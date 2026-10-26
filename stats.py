"""
stats.py — statistical report per topic (no LLM, runs in seconds).

Reads <topics>/threads.csv (from organize.py) and the posts .jsonl, and writes:
  <topics>/STATS.md   threads, comments, activity per year, keywords, most commented / top scored
  <topics>/stats.csv  one row per topic (for Excel / BI tools)

Example:
  python stats.py --sub selfhosted --topics topics
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from organize import STOP_EXTRA, ENGLISH_STOP_WORDS, clean


def read_posts(path):
    posts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
                posts[p["id"]] = p
            except (json.JSONDecodeError, KeyError):
                pass
    return posts


def bar(n, top, width=30):
    return "█" * max(1 if n else 0, round(n / top * width)) if top else ""


def main():
    ap = argparse.ArgumentParser(description="Per-topic statistics report.")
    ap.add_argument("--sub", help="subreddit name (reads <sub>_posts.jsonl)")
    ap.add_argument("--posts", help="explicit path to the posts .jsonl")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--topics", default="topics", help="output folder of organize.py")
    ap.add_argument("--top", type=int, default=5, help="threads listed per ranking")
    a = ap.parse_args()

    if not a.posts and not a.sub:
        found = sorted(Path(a.data_dir).glob("*_posts.jsonl"))
        if len(found) != 1:
            sys.exit("Use --sub NAME or --posts PATH")
        a.posts = str(found[0])
        a.sub = found[0].name[:-len("_posts.jsonl")]
    posts = read_posts(a.posts or Path(a.data_dir) / f"{a.sub}_posts.jsonl")
    root = Path(a.topics)

    threads = defaultdict(list)
    with open(root / "threads.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            p = posts.get(row["id"], {})
            row["n_comments"] = int(row["n_comments"])
            row["score"] = int(p.get("score") or 0)
            row["year"] = row["date"][:4]
            row["link"] = "https://www.reddit.com" + p.get("permalink", "")
            row["text"] = clean(f"{p.get('title', '')} {p.get('selftext', '')}").replace("'", "")
            threads[row["topic"]].append(row)

    names = sorted(threads, key=lambda t: -len(threads[t]))
    total_t = sum(len(v) for v in threads.values())
    total_c = sum(h["n_comments"] for v in threads.values() for h in v)

    stop = sorted(set(ENGLISH_STOP_WORDS) | set(STOP_EXTRA) | {(a.sub or "").lower()})
    vec = TfidfVectorizer(stop_words=stop, max_df=0.6, ngram_range=(1, 2),
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = vec.fit_transform([" ".join(h["text"] for h in threads[t]) for t in names])
    terms = vec.get_feature_names_out()
    kw = {t: [terms[j] for j in X[i].toarray().ravel().argsort()[::-1][:10]
              if X[i, j] > 0] for i, t in enumerate(names)}

    years = sorted({h["year"] for v in threads.values() for h in v})
    out = [f"# r/{a.sub}: statistics per topic", "",
           f"**{total_t} threads · {total_c} comments · {len(names)} topics · {years[0]}–{years[-1]}**", "",
           "## Overview", "",
           "| # | Topic | Threads | % | Comments | Comments/thread | From | To |",
           "|---|-------|--------:|--:|---------:|----------------:|------|----|"]
    rows = []
    for i, t in enumerate(names, 1):
        v = threads[t]
        nc = sum(h["n_comments"] for h in v)
        dates = sorted(h["date"] for h in v)
        out.append(f"| {i} | {t} | {len(v)} | {len(v) / total_t:.0%} | {nc} | {nc / len(v):.1f} | "
                   f"{dates[0][:10]} | {dates[-1][:10]} |")
        rows.append([t, len(v), nc, round(nc / len(v), 1), dates[0][:10], dates[-1][:10], ", ".join(kw[t])])

    per_year = Counter(h["year"] for v in threads.values() for h in v)
    m = max(per_year.values())
    out += ["", "## Activity per year", "", "```"]
    out += [f"{y}  {bar(per_year[y], m)} {per_year[y]}" for y in years]
    out += ["```", ""]

    for i, t in enumerate(names, 1):
        v = threads[t]
        nc = sum(h["n_comments"] for h in v)
        out += [f"## {i}. {t}", "",
                f"- **Threads:** {len(v)} ({len(v) / total_t:.0%}) · **Comments:** {nc}",
                f"- **Keywords:** {', '.join(kw[t])}", "", "Threads per year:", "", "```"]
        c = Counter(h["year"] for h in v)
        mx = max(c.values())
        out += [f"{y}  {bar(c[y], mx, 20)} {c[y]}" for y in years if c[y]]
        out += ["```", "", "Most commented:", ""]
        for h in sorted(v, key=lambda h: (-h["n_comments"], -h["score"]))[:a.top]:
            out.append(f"- [{h['title']}]({h['link']}) · {h['date'][:10]} · "
                       f"{h['n_comments']} comments · score {h['score']}")
        out += ["", "Top scored:", ""]
        for h in sorted(v, key=lambda h: (-h["score"], -h["n_comments"]))[:a.top]:
            out.append(f"- [{h['title']}]({h['link']}) · {h['date'][:10]} · "
                       f"score {h['score']} · {h['n_comments']} comments")
        out.append("")

    (root / "STATS.md").write_text("\n".join(out), encoding="utf-8")
    with open(root / "stats.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic", "threads", "comments", "comments_per_thread", "from", "to", "keywords"])
        w.writerows(rows)
    print(f"{len(names)} topics, {total_t} threads, {total_c} comments -> {root / 'STATS.md'}")


if __name__ == "__main__":
    main()
