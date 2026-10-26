"""Fast tests, no Ollama needed:  pip install pytest && pytest -q"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import organize  # noqa: E402
import summarize  # noqa: E402


def test_clean_removes_noise():
    t = organize.clean("See [docs](https://x.io/a) and https://foo.com u/someone r/sub **bold**")
    assert "http" not in t and "u/someone" not in t and "r/sub" not in t and "*" not in t
    assert organize.clean("[deleted]") == ""


def test_bot_detection():
    assert organize.is_bot({"author": "AutoModerator"})
    assert organize.is_bot({"author": "RemindMeBot"})
    assert not organize.is_bot({"author": "robot_fan"})


def test_merge_duplicates_joins_split_topics():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=64), rng.normal(size=64)
    # topic A artificially split in two labels (0 and 1), topic B is label 2
    emb = np.vstack([a + rng.normal(size=64) * 0.3 for _ in range(40)] +
                    [b + rng.normal(size=64) * 0.3 for _ in range(20)])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    labels, merged = organize.merge_duplicates(emb, labels.copy())
    assert merged == 1
    assert len(set(labels[:40])) == 1 and labels[0] != labels[-1]


def test_summarizer_always_converges():
    s = summarize.Summarizer("m", "http://x", 8192, 3000, "en")
    calls = []

    def fake_llm(prompt, max_tokens):
        calls.append(max_tokens)
        return "- point " * (max_tokens * 4 // 8)  # model that always writes up to its cap

    s.llm = fake_llm
    text = ("word " * 40 + "\n") * 5000
    level0 = len(s.chunks(text, 3000))
    out = s.summarize(text, "topic")
    # each level shrinks ~5x, so total calls ~ level0 * (1 + 1/5 + 1/25 + ...) ~ 1.25 * level0
    assert out and len(calls) <= 1.35 * level0 + 3


def test_summarizer_guard_against_non_shrinking_model():
    s = summarize.Summarizer("m", "http://x", 8192, 3000, "en")
    s.llm = lambda prompt, max_tokens: prompt  # worst case: echoes the whole input
    assert s.summarize(("word " * 40 + "\n") * 2000, "topic")


def test_chunks_cover_all_text():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(500))
    parts = summarize.Summarizer.chunks(text, 1000)
    assert all(len(p) <= 1100 for p in parts)
    assert "".join(parts).replace("\n", "") == text.replace("\n", "")
