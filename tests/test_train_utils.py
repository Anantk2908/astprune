import random

import numpy as np
import torch

from astprune.data.examples import TokenStore
from astprune.train import TrainConfig, epoch_order, filter_examples, info_nce, make_batch


def example(i, repo="a/b", n_gold=1):
    base = i * 100
    return {"instance_id": f"x-{i}", "repo": repo, "query": i, "pos": [base, base + 1],
            "neg_same": list(range(base + 10, base + 30)), "neg_import": list(range(base + 50, base + 60)),
            "n_gold_functions": n_gold}


def test_token_store_round_trip(tmp_path):
    arrays = [np.array([1, 2, 3], dtype=np.uint16), np.array([], dtype=np.uint16), np.array([9], dtype=np.uint16)]
    TokenStore.from_arrays(arrays).save(tmp_path / "t")
    store = TokenStore.load(tmp_path / "t")
    assert len(store) == 3
    assert [store[i].tolist() for i in range(3)] == [[1, 2, 3], [], [9]]


def test_filter_drops_patches_touching_many_functions():
    examples = [example(0, n_gold=1), example(1, n_gold=9)]
    assert [e["instance_id"] for e in filter_examples(examples, 5)] == ["x-0"]
    assert len(filter_examples(examples, 0)) == 2


def test_epoch_order_caps_big_repositories():
    examples = [example(i, repo="big/repo") for i in range(50)] + [example(100 + i, repo="small/repo") for i in range(5)]
    order = epoch_order(examples, max_per_repo=10, rng=random.Random(0))
    repos = [examples[i]["repo"] for i in order]
    assert repos.count("big/repo") == 10 and repos.count("small/repo") == 5
    assert len(set(order)) == len(order)
    other = epoch_order(examples, max_per_repo=10, rng=random.Random(1))
    assert set(order) != set(other)  # a different sample of the big repository each epoch


def test_make_batch_shapes_and_targets():
    cfg = TrainConfig(same_file_negatives=4, import_negatives=2)
    batch = [example(0), example(1), example(2)]
    node_ids, targets, ignore = make_batch(batch, cfg, random.Random(0))
    assert len(node_ids) == 3 * (1 + 4 + 2)
    assert targets == [0, 7, 14]
    for row, ex in enumerate(batch):
        assert node_ids[targets[row]] in ex["pos"]
    assert ignore.shape == (3, 21) and not ignore.any()


def test_make_batch_masks_other_gold_nodes_of_the_same_issue():
    cfg = TrainConfig(same_file_negatives=2, import_negatives=0)
    a = example(0)
    b = example(1)
    b["neg_same"] = [a["pos"][0], a["pos"][1], 999]  # another issue happens to bring a's gold nodes
    b["neg_same"] = b["neg_same"][:2]
    node_ids, targets, ignore = make_batch([a, b], cfg, random.Random(0))
    cols = [c for c, nid in enumerate(node_ids) if nid in a["pos"] and c != targets[0]]
    assert cols and all(ignore[0, c] for c in cols)
    assert not ignore[1].any()


def test_info_nce_prefers_the_right_node():
    torch.manual_seed(0)
    queries = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    noise = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    nodes = torch.cat([queries, noise])           # node i is identical to query i
    targets = torch.arange(4)
    ignore = torch.zeros(4, 12, dtype=torch.bool)
    good, acc = info_nce(queries, nodes, targets, ignore, temperature=0.05)
    bad, _ = info_nce(queries, nodes, (targets + 1) % 4, ignore, temperature=0.05)
    assert acc == 1.0 and good.item() < 0.1 < bad.item()


def test_info_nce_ignores_masked_columns():
    queries = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    nodes = torch.nn.functional.normalize(torch.tensor([[0.6, 0.8], [1.0, 0.0]]), dim=-1)
    targets = torch.tensor([0])
    unmasked, _ = info_nce(queries, nodes, targets, torch.tensor([[False, False]]), 0.05)
    masked, acc = info_nce(queries, nodes, targets, torch.tensor([[False, True]]), 0.05)
    assert masked.item() < 1e-4 < unmasked.item() and acc == 1.0
