"""
organize.py — turn a raw subreddit dump into full threads grouped by topic.

Input : <sub>_posts.jsonl and <sub>_comments.jsonl (Pushshift / Arctic Shift format)
Output: one folder per topic, one Markdown file per thread (post + full nested
        comment tree, nothing trimmed), plus INDEX.md and threads.csv.

Pipeline:
  1. Clean text (URLs, markdown, [deleted]/[removed], u/ and r/ mentions) and set bot comments aside.
  2. Embed each thread (Ollama or sentence-transformers).
  3. UMAP -> HDBSCAN ('leaf' selection) to find fine-grained topics; or KMeans with a fixed --k.
  4. Merge near-duplicate topics, then rescue loose threads into the nearest topic
     (both with relative similarity thresholds, so they work with any embedding model).
  5. Label topics with class-based TF-IDF (bigrams); optionally name them with a local LLM.
  6. Write everything and verify that every post and comment was accounted for.

Examples:
  python organize.py --sub selfhosted
  python organize.py --sub selfhosted --min-size 8
  python organize.py --sub selfhosted --k 20 --llm-names --lang es
"""

import argparse
import csv
import html
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    sys.exit("scikit-learn >= 1.3 is required:  pip install -U scikit-learn")

sys.setrecursionlimit(10000)

BOTS = {"automoderator", "remindmebot", "sneakpeekbot", "wikitextbot", "repostsleuthbot",
        "savevideo", "stabbot", "haikusbot", "linkifybot", "gifreversingbot", "timezone_bot"}

STOP_EXTRA = """
ive im id ill youre youve youll youd hes shes theyre theyve weve wed thats theres
whats whos lets dont doesnt didnt cant couldnt wont wouldnt shouldnt isnt arent
wasnt werent havent hasnt hadnt ve ll re don doesn didn just like really feel felt
know think thing things got get gets getting going gonna want wanted make makes
made way did does said say says lot lots actually pretty good bit sure maybe
probably also anyone someone something anything everyone people person use used
using time times day days https http www com reddit amp gt lt nbsp deleted removed
post posts comment comments thread sub subreddit edit thanks thank yes yeah
ok okay hey hi guys need help question user users
""".split()

MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
URL = re.compile(r"https?://\S+|www\.\S+|\b\S+\.(?:com|org|net|io|gov|edu)\S*")
MENTION = re.compile(r"(?<!\w)/?[ur]/[\w-]+")


# ------------------------------------------------------------------ helpers
def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def fmt_date(ts):
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(float(ts))))
    except (TypeError, ValueError):
        return "?"


def safe_name(text, length=60):
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text or "")
    text = re.sub(r"\s+", "_", text.strip())
    return text[:length].rstrip("._") or "untitled"


def slug(text):
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-") or "topic"


def is_bot(c):
    a = (c.get("author") or "").lower()
    return a in BOTS or (len(a) > 4 and a.endswith("bot"))


def clean(t):
    """Text used for clustering only. Thread files keep the original text."""
    if not t:
        return ""
    t = html.unescape(t)
    if t.strip() in ("[deleted]", "[removed]"):
        return ""
    t = MD_LINK.sub(r"\1", t)
    t = URL.sub(" ", t)
    t = MENTION.sub(" ", t)
    t = re.sub(r"[*_>#`~|]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def post_text(p):
    parts = [clean(p.get("title", "")), clean(p.get("selftext", ""))]
    for orig in p.get("crosspost_parent_list") or []:
        parts += [clean(orig.get("title", "")), clean(orig.get("selftext", ""))]
    return ". ".join(x for x in parts if x)


def embedding_doc(p, coms):
    """What the embedding model sees: the post; if it is short, add the top-scored comments."""
    base = post_text(p)
    if len(base) < 300:
        best = sorted(coms, key=lambda c: -to_int(c.get("score")))
        base = f"{base} {' '.join(clean(c.get('body')) for c in best[:5])}".strip()
    return base[:2000]


def keyword_doc(p, coms):
    t = post_text(p) + " " + " ".join(clean(c.get("body")) for c in coms)
    return t.replace("'", "").replace("’", "")


# ------------------------------------------------------------------ embeddings
def embed_ollama(docs, model, url, batch=32):
    print(f"Embedding {len(docs)} threads with Ollama ({model})...")
    endpoint = url.rstrip("/") + "/api/embed"
    prefix = "clustering: " if "nomic" in model else ""  # nomic expects a task prefix
    vecs = []
    for i in range(0, len(docs), batch):
        chunk = [prefix + (d or "(empty)") for d in docs[i:i + batch]]
        body = json.dumps({"model": model, "input": chunk, "truncate": True}).encode()
        req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                vecs += json.loads(r.read())["embeddings"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                sys.exit(f"Ollama does not have '{model}'. Run first:  ollama pull {model}")
            raise
        except urllib.error.URLError:
            sys.exit("Cannot reach Ollama. Open the Ollama app or run 'ollama serve'.")
        print(f"  {min(i + batch, len(docs))}/{len(docs)}", end="\r")
    print()
    emb = np.array(vecs, dtype=np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def embed(docs, model, ollama_url):
    if model.startswith("ollama:"):
        return embed_ollama(docs, model.split(":", 1)[1], ollama_url)
    from sentence_transformers import SentenceTransformer
    print(f"Embedding {len(docs)} threads with sentence-transformers ({model})...")
    return SentenceTransformer(model).encode(docs, batch_size=32, show_progress_bar=True,
                                             normalize_embeddings=True)


# ------------------------------------------------------------------ clustering
def reduce_dims(emb):
    try:
        import umap
        return umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                         metric="cosine", random_state=42).fit_transform(emb)
    except ImportError:
        print("  (umap-learn not installed, falling back to PCA; topics will be worse)")
        from sklearn.decomposition import PCA
        return PCA(n_components=10, random_state=42).fit_transform(emb)


def hdbscan(red, size):
    # 'leaf' keeps fine-grained subtopics. The default ('eom') tends to collapse a
    # single-community dataset into 2-3 giant clusters.
    return HDBSCAN(min_cluster_size=size, min_samples=min(5, size),
                   cluster_selection_method="leaf").fit_predict(red)


def centroids(emb, labels, valid):
    c = np.vstack([emb[labels == k].mean(axis=0) for k in valid])
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def own_similarity(emb, labels, valid, cents):
    pos = {c: j for j, c in enumerate(valid)}
    return np.array([cents[pos[l]] @ emb[i] for i, l in enumerate(labels) if l != -1])


def merge_duplicates(emb, labels, pct=50):
    """'leaf' over-splits: merge the closest pair of topics while their centroids are more
    similar than the median thread is to its own topic centroid. Relative, so model-agnostic."""
    merged = 0
    while True:
        valid = sorted(set(labels.tolist()) - {-1})
        if len(valid) <= 2:
            break
        cents = centroids(emb, labels, valid)
        threshold = np.percentile(own_similarity(emb, labels, valid, cents), pct)
        sim = cents @ cents.T
        np.fill_diagonal(sim, -1)
        i, j = np.unravel_index(sim.argmax(), sim.shape)
        if sim[i, j] < threshold:
            break
        labels[labels == valid[j]] = valid[i]
        merged += 1
    return labels, merged


def cluster(emb, min_size, k=None, rescue_pct=10, merge_pct=50):
    red = reduce_dims(emb)

    if k:
        from sklearn.cluster import KMeans
        print(f"Clustering into exactly {k} topics (KMeans on embeddings)...")
        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(red)

    print("Clustering (HDBSCAN). Diagnostic before merging, min size -> topics / unassigned:")
    for s in sorted({5, 8, 10, 15, 20, 30, min_size}):
        lab = hdbscan(red, s)
        mark = "  <- used" if s == min_size else ""
        print(f"    {s:>3} -> {len(set(lab.tolist()) - {-1}):>3} topics / {int((lab == -1).sum()):>5} unassigned{mark}")

    labels = hdbscan(red, min_size)
    if not set(labels.tolist()) - {-1}:
        sys.exit(f"No topics with min size {min_size}. Pick another value from the table above.")

    if merge_pct:
        before = len(set(labels.tolist()) - {-1})
        labels, merged = merge_duplicates(emb, labels, merge_pct)
        print(f"  merged near-duplicate topics: {before} -> {before - merged}")
    valid = sorted(set(labels.tolist()) - {-1})

    # Relative rescue threshold, model-agnostic: a loose thread joins its nearest topic
    # only if it is at least as similar as the least typical rescue_pct% of members.
    cents = centroids(emb, labels, valid)
    threshold = float(np.percentile(own_similarity(emb, labels, valid, cents), rescue_pct))

    loose = int((labels == -1).sum())
    rescued = 0
    for i in np.where(labels == -1)[0]:
        sims = cents @ emb[i]
        j = int(sims.argmax())
        if sims[j] >= threshold:
            labels[i] = valid[j]
            rescued += 1
    print(f"  {len(valid)} topics | {loose} unassigned -> {rescued} rescued "
          f"(threshold {threshold:.2f}), {loose - rescued} left in _unassigned")
    if len(valid) < 6:
        print("  ! Few topics. Pick a smaller min size from the table, or force a number: --k 20")
    return labels


def keywords(docs_by_topic, stopwords, n=6):
    """Class-based TF-IDF: each topic is one document, so terms that set it apart score high."""
    ids = list(docs_by_topic)
    vec = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 2), max_df=0.7,
                          sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = vec.fit_transform([docs_by_topic[k] for k in ids])
    terms = vec.get_feature_names_out()
    out = {}
    for i, k in enumerate(ids):
        row = X[i].toarray().ravel()
        chosen = []
        for j in row.argsort()[::-1]:
            if row[j] <= 0 or len(chosen) == n:
                break
            t = terms[j]
            if any(t in e or e in t for e in chosen):  # skip 'backup' if 'offsite backup' is in
                continue
            chosen.append(t)
        out[k] = chosen
    return out


def llm_name(kws, titles, model, url, lang):
    language = "Spanish" if lang == "es" else "English"
    prompt = ("Here are the keywords and some titles from a group of Reddit posts.\n"
              f"Keywords: {', '.join(kws)}\nTitles:\n- " + "\n- ".join(titles) + "\n\n"
              f"Reply ONLY with a short topic name in {language} (2 to 5 words), "
              "no quotes, no period, no explanation.")
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.2, "num_ctx": 4096}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            txt = json.loads(r.read())["response"]
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip().splitlines()[0]
        return txt.strip(" .\"'")[:60] or None
    except Exception as e:
        print(f"  ! LLM naming failed ({e}), using keywords")
        return None


# ------------------------------------------------------------------ rendering
def render_comments(items, out):
    children = defaultdict(list)
    ids = {c["id"] for c in items}
    roots = []
    for c in sorted(items, key=lambda c: to_int(c.get("created_utc"))):
        parent = c.get("parent_id") or ""
        if parent.startswith("t1_") and parent[3:] in ids:
            children[parent[3:]].append(c)
        else:
            roots.append(c)

    def write(c, depth):
        pre = "> " * depth
        out.append(f"{pre}**u/{c.get('author', '?')}** · {fmt_date(c.get('created_utc'))} · "
                   f"score {c.get('score', '?')}")
        out.append(pre)
        for line in (c.get("body") or "").splitlines() or [""]:
            out.append(f"{pre}{line}")
        out.append("")
        for ch in children[c["id"]]:
            write(ch, depth + 1)

    for c in roots:
        write(c, 0)


def render_thread(p, coms, sub):
    out = [f"# {p.get('title', '(untitled)')}", "",
           f"- Author: u/{p.get('author', '?')}",
           f"- Date: {fmt_date(p.get('created_utc'))}",
           f"- Score: {p.get('score', '?')} · Comments in archive: {len(coms)}",
           f"- Link: https://www.reddit.com{p.get('permalink', '')}"]
    if p.get("url") and f"reddit.com/r/{sub}" not in p.get("url", ""):
        out.append(f"- Linked URL: {p['url']}")
    out.append("")
    if p.get("selftext"):
        out += ["## Text", "", p["selftext"], ""]
    for orig in p.get("crosspost_parent_list") or []:
        out += [f"## Crosspost (r/{orig.get('subreddit', '?')}, u/{orig.get('author', '?')})", "",
                f"**{orig.get('title', '')}**", "", orig.get("selftext", "") or "(no text)", ""]
    out += ["## Comments", ""]
    if coms:
        render_comments(coms, out)
    else:
        out.append("(no comments)")
    return "\n".join(out)


# ------------------------------------------------------------------ main
def find_sub(data_dir):
    found = sorted(data_dir.glob("*_posts.jsonl"))
    if len(found) == 1:
        return found[0].name[:-len("_posts.jsonl")]
    names = ", ".join(f.name for f in found) or "none"
    sys.exit(f"Use --sub NAME. Found *_posts.jsonl files: {names}")


def main():
    ap = argparse.ArgumentParser(description="Group a subreddit dump into topic folders of full threads.")
    ap.add_argument("--sub", help="subreddit name (expects <sub>_posts.jsonl and <sub>_comments.jsonl)")
    ap.add_argument("--data-dir", default=".", help="folder with the .jsonl files (default: .)")
    ap.add_argument("--posts", help="explicit path to the posts .jsonl")
    ap.add_argument("--comments", help="explicit path to the comments .jsonl")
    ap.add_argument("--out", default="topics", help="output folder (default: topics)")
    ap.add_argument("--min-size", type=int, default=15, help="HDBSCAN minimum threads per topic")
    ap.add_argument("--k", type=int, help="force exactly K topics (KMeans) instead of HDBSCAN")
    ap.add_argument("--embed-model", default="ollama:nomic-embed-text",
                    help="'ollama:<model>' or a sentence-transformers model name/path")
    ap.add_argument("--rescue-pct", type=float, default=10, help="percentile for rescuing loose threads")
    ap.add_argument("--merge-pct", type=float, default=50,
                    help="merge topics whose centroids are closer than this percentile of thread-to-topic "
                         "similarity (lower = more merging, 0 = off)")
    ap.add_argument("--llm-names", action="store_true", help="name topics with a local LLM via Ollama")
    ap.add_argument("--llm-model", default="qwen3:4b-instruct-2507-q4_K_M")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--lang", choices=["en", "es"], default="en", help="language for LLM topic names")
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    sub = a.sub or (None if a.posts else find_sub(data_dir)) or "subreddit"
    posts_path = Path(a.posts) if a.posts else data_dir / f"{sub}_posts.jsonl"
    comments_path = Path(a.comments) if a.comments else data_dir / f"{sub}_comments.jsonl"
    out = Path(a.out)

    posts = read_jsonl(posts_path)
    comments = read_jsonl(comments_path) if comments_path.exists() else []
    if not posts:
        sys.exit(f"No posts found in {posts_path}")
    bots = [c for c in comments if is_bot(c)]
    humans = [c for c in comments if not is_bot(c)]
    print(f"r/{sub}: {len(posts)} posts, {len(comments)} comments ({len(bots)} from bots, set aside)")

    by_post = defaultdict(list)
    for c in humans:
        by_post[(c.get("link_id") or "").replace("t3_", "")].append(c)
    post_ids = {p["id"] for p in posts}

    emb = embed([embedding_doc(p, by_post.get(p["id"], [])) for p in posts], a.embed_model, a.ollama_url)
    labels = cluster(emb, a.min_size, a.k, a.rescue_pct, a.merge_pct)

    topic_ids = sorted(set(labels.tolist()) - {-1}, key=lambda k: -int((labels == k).sum()))
    docs = defaultdict(list)
    for p, l in zip(posts, labels):
        if l != -1:
            docs[l].append(keyword_doc(p, by_post.get(p["id"], [])))
    stop = sorted(set(ENGLISH_STOP_WORDS) | set(STOP_EXTRA) | {sub.lower()})
    kws = keywords({k: " ".join(docs[k]) for k in topic_ids}, stop)

    folder = {-1: "_unassigned"}
    for i, k in enumerate(topic_ids, 1):
        name = None
        if a.llm_names:
            members = [p for p, l in zip(posts, labels) if l == k]
            top = sorted(members, key=lambda p: -len(by_post.get(p["id"], [])))[:8]
            name = llm_name(kws[k], [p.get("title", "") for p in top], a.llm_model, a.ollama_url, a.lang)
            if name:
                print(f"  {i:02d}: {name}")
        folder[k] = f"{i:02d}_{slug(name or '-'.join(kws[k][:4]))[:50]}"
    kws[-1] = []

    out.mkdir(parents=True, exist_ok=True)
    rows, by_topic = [], defaultdict(list)
    written = 0
    for p, k in zip(posts, labels):
        coms = by_post.get(p["id"], [])
        written += len(coms)
        d = out / folder[k]
        d.mkdir(exist_ok=True)
        fname = f"{fmt_date(p.get('created_utc'))[:10]}_{p['id']}_{safe_name(p.get('title'))}.md"
        (d / fname).write_text(render_thread(p, coms, sub), encoding="utf-8")
        rows.append([p["id"], folder[k], fmt_date(p.get("created_utc")), p.get("title", ""), len(coms)])
        by_topic[folder[k]].append((fmt_date(p.get("created_utc"))[:10], p.get("title", ""), len(coms), fname))

    orphans = {k: v for k, v in by_post.items() if k not in post_ids}
    if orphans:
        d = out / "_orphan_comments"
        d.mkdir(exist_ok=True)
        for link, coms in orphans.items():
            lines = [f"# Comments of post {link} (post not in archive)", ""]
            render_comments(coms, lines)
            (d / f"{link}.md").write_text("\n".join(lines), encoding="utf-8")
            written += len(coms)

    if bots:
        d = out / "_bots"
        d.mkdir(exist_ok=True)
        lines = ["# Bot comments (excluded from analysis)", ""]
        for c in sorted(bots, key=lambda c: to_int(c.get("created_utc"))):
            lines += [f"**u/{c.get('author')}** · {fmt_date(c.get('created_utc'))} · "
                      f"post {(c.get('link_id') or '').replace('t3_', '')}", "", c.get("body") or "", ""]
        (d / "bots.md").write_text("\n".join(lines), encoding="utf-8")

    with open(out / "threads.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "topic", "date", "title", "n_comments"])
        w.writerows(sorted(rows, key=lambda r: (r[1], r[2])))

    kw_by_folder = {folder[k]: kws[k] for k in folder}
    mode = f"k = {a.k}" if a.k else f"min size {a.min_size}"
    index = [f"# r/{sub}: topic index", "",
             f"{len(posts)} threads · {len(humans)} human comments · {len(topic_ids)} topics · {mode}", ""]
    for topic in sorted(by_topic):
        threads = sorted(by_topic[topic])
        index += [f"## {topic} ({len(threads)} threads)", ""]
        if kw_by_folder.get(topic):
            index += [f"*Keywords:* {', '.join(kw_by_folder[topic])}", ""]
        index += [f"- {d} · {t} ({n} comments) → `{topic}/{fn}`" for d, t, n, fn in threads]
        index.append("")
    if orphans:
        index += [f"## _orphan_comments ({len(orphans)} files)", ""]
    if bots:
        index += [f"## _bots ({len(bots)} comments)", ""]
    (out / "INDEX.md").write_text("\n".join(index), encoding="utf-8")

    print("\n================ TOPICS ================")
    for k in topic_ids + [-1]:
        n = int((labels == k).sum())
        if n:
            print(f"{folder[k]:<54} {n:>5} threads  {', '.join(kws[k][:5])}")

    print("\n============= VERIFICATION =============")
    print(f"Posts read:    {len(posts)}  | threads written: {len(rows)}")
    print(f"Comments read: {len(comments)}  | written: {written} + bots: {len(bots)}")
    ok = len(posts) == len(rows) and len(comments) == written + len(bots)
    print("Everything accounted for." if ok else "WARNING: counts do not match.")
    print(f"Output: {out.resolve()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
