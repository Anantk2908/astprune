"""Contrastive fine-tuning of the node ranker.

Each step takes a few issues. Every issue brings one node the gold patch
changed (the positive) and a handful of nodes it did not change, mostly from
the same file. The loss (InfoNCE) asks the issue embedding to be closer to its
positive than to every other node in the batch, including the other issues'
nodes. Since only one to three nodes out of dozens are positive per issue,
this ranking loss fits better than a per-node yes/no classifier.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .data.examples import TokenStore
from .evaluation import FileLoader, candidate_paths
from .metrics import evaluate_instance, original_tokens
from .pruners import ASTPruner
from .scorers.encoder import Encoder, EncoderNodeScorer


@dataclass
class TrainConfig:
    run_name: str = "ranker"
    base_model: str = "microsoft/unixcoder-base"
    seed: int = 0
    epochs: float = 2.0
    batch_issues: int = 8
    same_file_negatives: int = 8
    import_negatives: int = 3
    accumulate: int = 2              # optimizer step every N batches
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    temperature: float = 0.05
    max_grad_norm: float = 1.0
    max_gold_functions: int = 5      # drop noisy patches that touch more functions; 0 keeps all
    max_per_repo: int = 1500         # per epoch, so one huge repository does not dominate
    evals_per_epoch: int = 2
    dev_setting: str = "oracle_imports"
    dev_budgets: tuple[float, ...] = (0.10, 0.25)
    amp: bool = True
    # Recomputes activations in the backward pass instead of storing them. On an
    # 8 GB card a batch of 96 nodes needs 12 GB without it and 3.7 GB with it.
    grad_checkpointing: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filter_examples(examples: list[dict], max_gold_functions: int) -> list[dict]:
    if max_gold_functions <= 0:
        return examples
    return [e for e in examples if e["n_gold_functions"] <= max_gold_functions]


def epoch_order(examples: list[dict], max_per_repo: int, rng: random.Random) -> list[int]:
    """Indices for one epoch: a fresh random sample of at most ``max_per_repo`` per repository."""
    by_repo: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        by_repo[ex["repo"]].append(i)
    order: list[int] = []
    for repo in sorted(by_repo):
        idx = by_repo[repo]
        rng.shuffle(idx)
        order.extend(idx[:max_per_repo] if max_per_repo > 0 else idx)
    rng.shuffle(order)
    return order


def make_batch(batch: list[dict], cfg: TrainConfig, rng: random.Random):
    """Pick one positive and some negatives per issue.

    Returns node text ids, the column of each issue's positive, and a mask of
    columns that must not count as negatives for an issue (its other gold
    nodes, or a duplicate of its positive brought in by another issue).
    """
    node_ids: list[int] = []
    targets: list[int] = []
    for ex in batch:
        targets.append(len(node_ids))
        node_ids.append(rng.choice(ex["pos"]))
        node_ids.extend(rng.sample(ex["neg_same"], min(cfg.same_file_negatives, len(ex["neg_same"]))))
        node_ids.extend(rng.sample(ex["neg_import"], min(cfg.import_negatives, len(ex["neg_import"]))))
    ignore = torch.zeros((len(batch), len(node_ids)), dtype=torch.bool)
    for row, ex in enumerate(batch):
        gold = set(ex["pos"])
        for col, nid in enumerate(node_ids):
            if nid in gold and col != targets[row]:
                ignore[row, col] = True
    return node_ids, targets, ignore


def info_nce(queries: torch.Tensor, nodes: torch.Tensor, targets: torch.Tensor,
             ignore: torch.Tensor, temperature: float) -> tuple[torch.Tensor, float]:
    """InfoNCE over unit vectors. Also returns how often the positive ranked first."""
    logits = queries @ nodes.T / temperature
    logits = logits.masked_fill(ignore, float("-inf"))
    loss = torch.nn.functional.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return loss, accuracy


@torch.no_grad()
def dev_recall(encoder: Encoder, records: list[dict], setting: str, budgets: tuple[float, ...],
               loader: FileLoader | None = None) -> dict:
    """Gold-function recall of the current weights on held-out issues, through the real pipeline."""
    was_training = encoder.model.training
    encoder.model.eval()
    encoder.clear_cache()
    method = ASTPruner(EncoderNodeScorer(encoder, name="dev"))
    own_loader = loader is None
    loader = loader or FileLoader()
    per_budget: dict[float, list[float]] = {b: [] for b in budgets}
    for rec in sorted(records, key=lambda r: (r["repo"], r["base_commit"])):
        if not rec["gold_functions"]:
            continue
        files = loader.load(rec, candidate_paths(rec, setting), {})
        if not files:
            continue
        scores = method.prepare(rec["issue"], files)
        total = original_tokens(files)
        for b in budgets:
            ctx = method.prune(scores, files, int(b * total))
            per_budget[b].append(evaluate_instance(files, rec["gold_functions"], {}, ctx)["gold_fn_recall"])
    if own_loader:
        loader.close()
    encoder.clear_cache()
    encoder.model.train(was_training)
    out = {f"recall_r{b:.2f}": float(np.mean(v)) for b, v in per_budget.items()}
    out["mean_recall"] = float(np.mean(list(out.values())))
    out["n"] = len(next(iter(per_budget.values())))
    return out


def train(cfg: TrainConfig, examples: list[dict], nodes: TokenStore, queries: TokenStore,
          dev_records: list[dict], out_dir: Path, resume: bool = False) -> dict:
    """Fine-tune and keep the checkpoint with the best dev recall in ``out_dir/best``.

    The full training state is saved after every dev check. With ``resume`` an
    interrupted run continues from its last check.
    """
    set_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = filter_examples(examples, cfg.max_gold_functions)

    encoder = Encoder(cfg.base_model, name="training")
    model = encoder.model
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    device = encoder.device
    use_amp = cfg.amp and device == "cuda"

    steps_per_epoch = len(epoch_order(examples, cfg.max_per_repo, random.Random(0))) // cfg.batch_issues
    if steps_per_epoch == 0:
        raise ValueError("not enough training examples for one batch")
    total_batches = int(steps_per_epoch * cfg.epochs)
    total_updates = max(1, total_batches // cfg.accumulate)
    warmup = max(1, int(total_updates * cfg.warmup_fraction))
    eval_every = max(cfg.accumulate, steps_per_epoch // cfg.evals_per_epoch // cfg.accumulate * cfg.accumulate)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim < 2 or "LayerNorm" in name else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda u: (u + 1) / warmup if u < warmup else max(0.0, (total_updates - u) / max(1, total_updates - warmup)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loader = FileLoader()
    history: list[dict] = []
    best = {"mean_recall": -1.0}
    step = 0
    minutes_before = 0.0
    state_path = out_dir / "state.pt"
    if resume and state_path.exists():
        # Pick up after the last dev check of an interrupted run.
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, history, best, minutes_before = state["step"], state["history"], state["best"], state["minutes"]
        print(f"  resumed from step {step} ({minutes_before:.1f} training minutes so far)", flush=True)
    log = (out_dir / "train_log.jsonl").open("a" if step else "w")
    started = time.time()

    def minutes() -> float:
        return minutes_before + (time.time() - started) / 60

    def evaluate_and_save(step: int) -> None:
        nonlocal best
        metrics = dev_recall(encoder, dev_records, cfg.dev_setting, cfg.dev_budgets, loader)
        metrics.update(step=step, epoch=round(step / steps_per_epoch, 3))
        history.append(metrics)
        print(f"  dev @ step {step}: {json.dumps(metrics)}", flush=True)
        if metrics["mean_recall"] > best["mean_recall"]:
            best = metrics
            model.save_pretrained(out_dir / "best")
            encoder.tokenizer.save_pretrained(out_dir / "best")
        tmp = state_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "history": history, "best": best, "minutes": minutes()}, tmp)
        tmp.replace(state_path)  # never leave a half-written state file behind

    if step == 0:
        evaluate_and_save(0)  # zero-shot reference with the same code path

    running_loss, running_acc, seen = 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    order: list[int] = []
    order_epoch = -1
    while step < total_batches:
        # The order of an epoch and the negatives of a step depend only on the
        # seed and their index, so a resumed run continues exactly where it was.
        epoch, position = divmod(step, steps_per_epoch)
        if epoch != order_epoch:
            order = epoch_order(examples, cfg.max_per_repo, random.Random(f"{cfg.seed}:epoch:{epoch}"))
            order_epoch = epoch
        if True:
            start = position * cfg.batch_issues
            batch = [examples[i] for i in order[start : start + cfg.batch_issues]]
            node_ids, targets, ignore = make_batch(batch, cfg, random.Random(f"{cfg.seed}:step:{step}"))
            node_in = encoder.pad([nodes[i].astype(np.int64).tolist() for i in node_ids])
            query_in = encoder.pad([queries[ex["query"]].astype(np.int64).tolist() for ex in batch])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                node_vecs = encoder.forward(*node_in)
                query_vecs = encoder.forward(*query_in)
            loss, acc = info_nce(query_vecs, node_vecs, torch.tensor(targets, device=device),
                                 ignore.to(device), cfg.temperature)
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"loss became {loss.item()} at step {step}")
            scaler.scale(loss / cfg.accumulate).backward()
            step += 1
            if step % cfg.accumulate == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running_loss += loss.item()
            running_acc += acc
            seen += 1
            if step % 50 == 0:
                row = {"step": step, "loss": round(running_loss / seen, 4), "top1": round(running_acc / seen, 4),
                       "lr": scheduler.get_last_lr()[0], "minutes": round(minutes(), 2),
                       "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else 0}
                log.write(json.dumps(row) + "\n")
                log.flush()
                if step % 250 == 0:
                    print(f"  step {step}/{total_batches} {json.dumps(row)}", flush=True)
                running_loss, running_acc, seen = 0.0, 0.0, 0
            if step % eval_every == 0 or step == total_batches:
                evaluate_and_save(step)
    log.close()
    loader.close()
    model.save_pretrained(out_dir / "last")
    encoder.tokenizer.save_pretrained(out_dir / "last")

    summary = {
        "config": asdict(cfg), "examples_used": len(examples), "steps_per_epoch": steps_per_epoch,
        "total_batches": total_batches, "train_minutes": round(minutes(), 1),
        "parameters": encoder.n_parameters, "best": best, "dev_history": history,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "torch": torch.__version__,
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
