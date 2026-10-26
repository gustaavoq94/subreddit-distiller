# 🧪 subreddit-distiller

**Turn thousands of Reddit threads into clean topics and readable summaries, 100% on your own machine.**

Point it at a subreddit dump. It groups every thread by *meaning* (not keywords), keeps each conversation intact with its full comment tree, writes per-topic statistics, and has a small local LLM write a structured summary of each topic. No API keys, no cloud, and your data never leaves your computer.

Built and tested with a 4B-parameter model (Qwen3 4B) on a consumer laptop.

```
r/selfhosted dump (1.4k posts, 10k comments)
        │
        ▼  organize.py   embeddings → UMAP → HDBSCAN → merge duplicates → c-TF-IDF labels
topics/
 ├── 01_wireguard-tailscale-tunnel-remote/   ← one .md per thread, full nested comments
 ├── 02_jellyfin-transcoding-subtitles/
 ├── ...
 ├── INDEX.md   threads.csv   STATS.md        ← stats.py
        │
        ▼  summarize.py  hierarchical map-reduce with a local LLM
summaries/
 ├── 01_wireguard-tailscale-tunnel-remote.md  ← overview · main ideas · consensus · debates · resources
 └── _ALL.md
```

## Why

Reading a whole community to figure out *what people actually say* takes days. Generic "summarize this" prompts break on long text, cloud LLMs mean sending your data out, and keyword-based topic modeling splits one topic into five clusters full of URLs and bot messages. This repo is the pipeline I wish I'd had: honest about its limits, verifiable, and cheap to run.

Use it for **market and product research** (pain points, feature requests), **community management**, **UX research**, **academic studies of online communities**, or just to catch up on a niche hobby.

## Highlights

- **Topics by meaning.** Sentence embeddings + UMAP + HDBSCAN, so "can't reach my server from outside" and "VPN tunnel keeps dropping" land together. The number of topics comes from the data, or you can force it with `--k`.
- **Fixes the classic HDBSCAN failure.** A single-community dataset tends to collapse into 2-3 giant clusters. This uses `leaf` selection and then **automatically merges near-duplicate topics** with a relative similarity threshold. On the synthetic benchmark this raised agreement with the true topics (ARI) from 0.58 to 0.93.
- **Model-agnostic thresholds.** Rescuing loose threads and merging topics use percentiles of the data's own similarity distribution, not magic cosine numbers, so switching embedding models doesn't break them.
- **Cleans before it clusters.** Strips URLs, markdown, `u/` and `r/` mentions and `[deleted]`/`[removed]`. Bot comments (AutoModerator & co.) are set aside, not lost.
- **Nothing gets lost.** Every post and comment ends up in exactly one file (topic, `_unassigned`, `_orphan_comments` or `_bots`), and the script verifies the counts at the end.
- **Summaries that converge.** Map-reduce with a capped output per call means each level shrinks the text about 5x, and a convergence guard makes infinite loops impossible. It is resumable: stop it any time, and finished topics are kept.
- **Runs on small hardware.** Embeddings and summaries both go through [Ollama](https://ollama.com). sentence-transformers is optional.
- **English or Spanish** output (`--lang en|es`).

## Quickstart

```bash
git clone https://github.com/gustaavoq94/subreddit-distiller
cd subreddit-distiller
pip install -r requirements.txt

ollama pull nomic-embed-text                 # embeddings (~270 MB)
ollama pull qwen3:4b-instruct-2507-q4_K_M    # summaries (~2.5 GB), or any model you like

# try it with synthetic demo data
python examples/make_demo_data.py
python organize.py  --sub demo_homelab --data-dir examples/demo --out topics
python stats.py     --sub demo_homelab --data-dir examples/demo --topics topics
python summarize.py --topics topics --out summaries
```

### With real data

You need two files in the standard Reddit/Pushshift JSON-lines format:

```
<sub>_posts.jsonl      one submission per line (id, title, selftext, author, created_utc, score, permalink…)
<sub>_comments.jsonl   one comment per line   (id, link_id, parent_id, body, author, created_utc, score…)
```

[Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) produces exactly this for a single subreddit, and Pushshift-style dumps work too. Please respect Reddit's terms and the privacy of the people who wrote those threads.

```bash
python organize.py --sub selfhosted --data-dir data
```

## Tuning the topics

`organize.py` prints a diagnostic table before writing anything:

```
Clustering (HDBSCAN). Diagnostic before merging, min size -> topics / unassigned:
      5 ->  42 topics /   223 unassigned
     10 ->  22 topics /   174 unassigned
     15 ->  14 topics /   133 unassigned  <- used
     30 ->  10 topics /    25 unassigned
  merged near-duplicate topics: 14 -> 8
  8 topics | 133 unassigned -> 117 rescued (threshold 0.88), 16 left in _unassigned

================ TOPICS ================
01_jellyfin-transcoding-streaming-subtitles    134 threads  jellyfin, transcoding, streaming, subtitles, library
02_wireguard-remote-tunnel-access              102 threads  wireguard, remote, tunnel, access, tailscale
03_hypervisor-cluster-proxmox-virtual           99 threads  hypervisor, cluster, proxmox, virtual, passthrough
...
============= VERIFICATION =============
Posts read:    638  | threads written: 638
Comments read: 1978 | written: 1977 + bots: 1
Everything accounted for.
```

| Want | Do |
|---|---|
| more, finer topics | `--min-size 8` or `--merge-pct 75` |
| fewer, broader topics | `--min-size 25` or `--merge-pct 25` |
| an exact number | `--k 20` |
| readable topic names | `--llm-names` (add `--lang es` for Spanish) |
| a different embedding model | `--embed-model ollama:mxbai-embed-large` or `--embed-model all-mpnet-base-v2` (sentence-transformers) |

Delete the output folder before re-running with different settings, since topic folder names change.

## Summaries

```bash
python summarize.py --topics topics --out summaries --lang en --model qwen3:4b-instruct-2507-q4_K_M
```

Each summary has the same structure: **Overview · Main ideas · Consensus and most repeated opinions · Disagreements and debates · Resources mentioned**. `summarize.py` works on *any* folder of subfolders containing `.md .txt .json .jsonl .csv .html`, not only on this repo's output.

**Speed reference:** about 20 s per 12k-character chunk with Qwen3 4B on a laptop GPU, so roughly 4-5 h for 1.4k threads plus 10k comments. Run it overnight. If you are short on VRAM, lower `--ctx 6144 --chunk-chars 8000`.

## How it works

1. **Clean**: URLs, markdown, mentions and deleted bodies are removed *for clustering only*. Thread files keep the original text.
2. **Embed**: each thread becomes a vector built from its post, plus its top-scored comments if the post is short.
3. **Reduce + cluster**: UMAP (5D, cosine), then HDBSCAN with `leaf` selection for fine-grained topics.
4. **Merge**: repeatedly join the two closest topics while their centroids are more similar than the median thread is to its own topic.
5. **Rescue**: an unassigned thread joins its nearest topic if it is at least as similar as the least typical 10% of that topic's members.
6. **Label**: class-based TF-IDF (each topic = one document, unigrams + bigrams, chat stopwords removed). Optionally, a local LLM names each topic.
7. **Summarize**: chunk → bullet summaries (capped at about 1/5 of the input) → merge level by level → final structured summary.

## Tests

```bash
pip install pytest && pytest -q
```

They cover text cleaning, bot detection, duplicate-topic merging, chunking, and convergence of the summarizer, including a worst-case model that never shortens its input. No Ollama needed.

## Limitations

- Topic quality depends on the embedding model. `nomic-embed-text` is a good, light default for English; for other languages try a multilingual model.
- Small LLMs can still miss nuance or over-generalize. Summaries point you to where to read, not to the final truth. Every thread is kept in full so you can check.
- Reddit dumps can contain personal data. Keep outputs private unless you have the right to share them.

---

## 🇪🇸 En español

**Convierte miles de hilos de Reddit en temas limpios y resúmenes legibles, todo en tu propia máquina.**

Agrupa cada hilo por *significado* (no por palabras clave), guarda las conversaciones completas, genera estadísticas por tema y usa un LLM local pequeño (vía Ollama) para escribir un resumen estructurado de cada tema. No necesita API keys ni nube.

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct-2507-q4_K_M
python organize.py  --sub NOMBRE --data-dir datos --llm-names --lang es
python stats.py     --sub NOMBRE --data-dir datos --topics topics
python summarize.py --topics topics --lang es
```

Si salen muy pocos temas, baja `--min-size` o fija el número con `--k 20`. Si salen demasiados, súbelo. El resumen se puede cortar en cualquier momento y retoma donde quedó.

## License

MIT © 2026 Gustavo Quevedo
