"""
summarize.py — summarize every topic folder with a local LLM (Ollama), 100% offline.

Works on the output of organize.py, but also on ANY folder of subfolders with text
(.md .txt .json .jsonl .csv .html): each subfolder is one topic.

Long topics are handled with hierarchical map-reduce:
  - split into chunks -> summarize each chunk (capped output, so every level compresses)
  - merge partial summaries level by level until one final structured summary is left
  - a convergence guard makes infinite loops impossible

Resumable: finished topics are saved immediately and skipped on the next run.

Examples:
  python summarize.py --topics topics
  python summarize.py --topics topics --lang es --model qwen3:4b-instruct-2507-q4_K_M
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EXT = {".txt", ".md", ".json", ".jsonl", ".csv", ".html"}
JSON_FIELDS = ("title", "selftext", "body", "text", "comment", "content")
MD_META = ("**u/", "- Author:", "- Date:", "- Score:", "- Link:", "- Linked URL:")

PROMPTS = {
    "en": {
        "chunk": ("Summarize in English, as short bullet points, the main ideas and opinions in this "
                  "Reddit excerpt about '{topic}'. Ignore greetings, spam and off-topic messages. "
                  "Maximum 15 bullets. Do not invent anything.\n\n{text}"),
        "merge": ("These are partial summaries about '{topic}'. Merge them into ONE bullet list in "
                  "English: combine repeated ideas, remove duplicates and keep what is mentioned most. "
                  "Maximum 15 bullets. Do not invent anything.\n\n{text}"),
        "final": ("You are an analyst. Below is Reddit content (or partial summaries) about the topic "
                  "'{topic}'. Write a final summary in English with this format:\n\n"
                  "## Overview\n(3-5 lines)\n## Main ideas\n(bullets)\n"
                  "## Consensus and most repeated opinions\n(bullets)\n"
                  "## Disagreements and debates\n(bullets)\n"
                  "## Resources and references mentioned\n(bullets; only if any)\n\n"
                  "Do not invent anything that is not in the text.\n\nCONTENT:\n{text}"),
        "all": "All summaries",
    },
    "es": {
        "chunk": ("Resume en español, en viñetas breves, las ideas y opiniones principales de este "
                  "fragmento de Reddit sobre '{topic}'. Ignora saludos, spam y mensajes irrelevantes. "
                  "Máximo 15 viñetas. No inventes.\n\n{text}"),
        "merge": ("Estos son resúmenes parciales sobre '{topic}'. Fusiónalos en UNA sola lista de "
                  "viñetas en español: junta las ideas repetidas, quita duplicados y conserva lo más "
                  "mencionado. Máximo 15 viñetas. No inventes.\n\n{text}"),
        "final": ("Eres un analista. Abajo hay contenido de Reddit (o resúmenes parciales) sobre el tema "
                  "'{topic}'. Escribe en español un resumen final con este formato:\n\n"
                  "## Resumen general\n(3-5 líneas)\n## Ideas principales\n(viñetas)\n"
                  "## Consensos y opiniones más repetidas\n(viñetas)\n"
                  "## Desacuerdos o debates\n(viñetas)\n"
                  "## Recursos o referencias mencionadas\n(viñetas; solo si aparecen)\n\n"
                  "No inventes nada que no esté en el texto.\n\nCONTENIDO:\n{text}"),
        "all": "Todos los resúmenes",
    },
}

csv.field_size_limit(2**31 - 1)


# ------------------------------------------------------------------ reading
def _walk_json(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in JSON_FIELDS and isinstance(v, str):
                v = v.strip()
                if v and v not in ("[deleted]", "[removed]"):
                    out.append(v)
            else:
                _walk_json(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _walk_json(x, out)


def read_text(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"   ! cannot read {path.name}: {e}")
        return ""
    if ext in (".json", ".jsonl"):
        out = []
        try:
            _walk_json(json.loads(raw), out)
        except json.JSONDecodeError:
            for line in raw.splitlines():
                try:
                    _walk_json(json.loads(line), out)
                except json.JSONDecodeError:
                    pass
        return "\n\n".join(out)
    if ext == ".csv":
        return "\n".join(" | ".join(c for c in row if c) for row in csv.reader(raw.splitlines()))
    if ext == ".html":
        return re.sub(r"<[^>]+>", " ", raw)
    if ext == ".md":  # thread files from organize.py: drop metadata and deleted bodies
        keep = []
        for line in raw.splitlines():
            s = line.lstrip("> ").strip()
            if s and s not in ("[deleted]", "[removed]") and not s.startswith(MD_META):
                keep.append(s)
        return "\n".join(keep)
    return raw


def topics(root: Path):
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
            yield d.name, files


# ------------------------------------------------------------------ LLM
class Summarizer:
    def __init__(self, model, url, ctx, chunk_chars, lang, final_tokens=1500):
        self.model, self.url, self.ctx = model, url.rstrip("/") + "/api/generate", ctx
        self.chunk_chars, self.p = chunk_chars, PROMPTS[lang]
        # ~4 chars per token; cap each partial at ~1/5 of its input so every level shrinks ~5x
        self.partial_tokens = max(150, chunk_chars // 20)
        self.final_tokens = final_tokens

    def llm(self, prompt, max_tokens):
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"num_ctx": self.ctx, "temperature": 0.3,
                                       "num_predict": max_tokens}}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    txt = json.loads(r.read())["response"]
                return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    sys.exit(f"Ollama does not have '{self.model}'. Check the name with: ollama list")
                if attempt == 2:
                    raise
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"   ! LLM call failed ({e}), retrying in 10s...")
                time.sleep(10)

    @staticmethod
    def chunks(text, n):
        cur, out = "", []
        for para in text.split("\n"):
            if len(cur) + len(para) > n and cur:
                out.append(cur)
                cur = ""
            while len(para) > n:
                out.append(para[:n])
                para = para[n:]
            cur += para + "\n"
        if cur.strip():
            out.append(cur)
        return out

    def summarize(self, text, topic, level=0, prev_chunks=None):
        if len(text) <= self.chunk_chars:
            return self.llm(self.p["final"].format(topic=topic, text=text), self.final_tokens)

        parts = self.chunks(text, self.chunk_chars)
        pad = "  " * level
        print(f"   {pad}level {level}: {len(text):,} chars -> {len(parts)} chunks")

        # Convergence guard: if this level did not shrink, trim proportionally and finish.
        if prev_chunks is not None and len(parts) >= prev_chunks:
            print(f"   {pad}! not converging, trimming and finishing this topic")
            share = self.chunk_chars // len(parts)
            return self.summarize("\n".join(x[:share] for x in parts), topic, level + 1)

        template = self.p["chunk"] if level == 0 else self.p["merge"]
        partials, t0 = [], time.time()
        for i, chunk in enumerate(parts, 1):
            partials.append(self.llm(template.format(topic=topic, text=chunk), self.partial_tokens))
            eta = (time.time() - t0) / i * (len(parts) - i) / 60
            print(f"   {pad}chunk {i}/{len(parts)}  (~{eta:.0f} min left in this level)")
        return self.summarize("\n\n".join(partials), topic, level + 1, prev_chunks=len(parts))


# ------------------------------------------------------------------ main
def preflight(url, model):
    """Fail fast if Ollama is down or the model is missing, instead of retrying per topic."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=10) as r:
            names = {m.get("name", "") for m in json.loads(r.read()).get("models", [])}
    except Exception:
        sys.exit(f"Cannot reach Ollama at {url}. Open the Ollama app or run 'ollama serve'.")
    if names and model not in names and f"{model}:latest" not in names:
        sys.exit(f"Ollama does not have '{model}'. Available: {', '.join(sorted(names))}\n"
                 f"Pass one with --model, or run: ollama pull {model}")


def main():
    ap = argparse.ArgumentParser(description="Summarize each topic folder with a local LLM via Ollama.")
    ap.add_argument("--topics", default="topics", help="folder with one subfolder per topic")
    ap.add_argument("--out", default="summaries", help="output folder (default: summaries)")
    ap.add_argument("--model", default="qwen3:4b-instruct-2507-q4_K_M", help="Ollama model name")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--lang", choices=["en", "es"], default="en")
    ap.add_argument("--chunk-chars", type=int, default=12000, help="characters per chunk")
    ap.add_argument("--ctx", type=int, default=8192, help="model context (num_ctx)")
    ap.add_argument("--only", nargs="*", help="only these topic folders")
    a = ap.parse_args()

    root, out = Path(a.topics), Path(a.out)
    if not root.is_dir():
        sys.exit(f"Folder not found: {root}")
    preflight(a.ollama_url, a.model)
    out.mkdir(parents=True, exist_ok=True)
    s = Summarizer(a.model, a.ollama_url, a.ctx, a.chunk_chars, a.lang)

    todo = [t for t in topics(root) if not a.only or t[0] in a.only]
    print(f"{len(todo)} topics. Model: {a.model}\n")
    start = time.time()
    for i, (topic, files) in enumerate(todo, 1):
        dest = out / f"{topic}.md"
        if dest.exists():
            print(f"[{i}/{len(todo)}] {topic}: already done, skipping")
            continue
        text = "\n\n".join(t for t in (read_text(f) for f in files) if t.strip())
        if not text.strip():
            print(f"[{i}/{len(todo)}] {topic}: no text, skipping")
            continue
        print(f"[{i}/{len(todo)}] {topic}: {len(files)} files, {len(text):,} chars")
        t0 = time.time()
        try:
            summary = s.summarize(text, topic)
        except Exception as e:
            print(f"   ! error in {topic}: {e}")
            continue
        dest.write_text(f"# {topic}\n\n{summary}\n", encoding="utf-8")
        print(f"   done in {(time.time() - t0) / 60:.1f} min -> {dest.name}\n")

    done = sorted(p for p in out.glob("*.md") if not p.name.startswith("_"))
    (out / "_ALL.md").write_text(f"# {PROMPTS[a.lang]['all']}\n\n" + "\n\n---\n\n".join(
        p.read_text(encoding="utf-8") for p in done), encoding="utf-8")
    print(f"\nFinished in {(time.time() - start) / 60:.0f} min. {len(done)} summaries in {out} (+ _ALL.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
