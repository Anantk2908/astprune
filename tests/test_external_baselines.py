"""Tests for the parts of the external baseline adapters that need no model."""

from astprune.baselines.longcodezip import LongCodeZipBaseline, lines_of_output, locate_chunks
from astprune.parse import parse_source


def test_chunks_are_located_by_line(sample_source):
    pf = parse_source(sample_source, "a.py")
    lines = pf.lines
    chunk_a = "\n".join(lines[7:11])       # def top_level ... return total
    chunk_b = "\n".join(lines[13:21])      # @decorator ... return inner() + y
    spans = locate_chunks(pf, [chunk_a, "\n" + chunk_b + "\n", "not in the file at all"])
    assert spans == [(8, 11), (14, 21), None]


def test_output_lines_map_back_in_order(sample_source):
    pf = parse_source(sample_source, "a.py")
    text = "\n".join([pf.lines[7], pf.lines[8], "# ... ", pf.lines[10], "", pf.lines[28]])
    assert lines_of_output(pf, text) == {8, 9, 11, 29}


def test_repeated_lines_are_matched_forward_only():
    pf = parse_source("x = 1\ny = 2\nx = 1\n", "a.py")
    assert lines_of_output(pf, "y = 2\nx = 1") == {2, 3}


def test_unknown_mode_is_rejected():
    try:
        LongCodeZipBaseline(mode="other")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
