#!/usr/bin/env python3
"""Generate a labeled AI-code corpus from latest frontier models via OpenRouter (ground truth
for the 'do AI models have code fingerprints?' experiment). Identical prompts across models =
controlled: the only variable is the model. Tracks cost live and ABORTS at a hard cap.

Usage:  python3 gen_ai_corpus.py [N_TASKS]   (default = all; use 2 for a pilot)
Key is read from the gitignored scratchpad file (never committed).
"""
import json
import os
import sys
import time
import urllib.request

SCRATCH = "/private/tmp/claude-501/-Users-tristenharr-duhclaudeknowsyou/4d4c9584-9d59-4b78-a64a-36a229af2782/scratchpad"
KEY = open(os.path.join(SCRATCH, ".orkey")).read().strip()
OUT = os.path.join(SCRATCH, "ai_corpus.json")
COST_CAP = 1.50  # hard abort — keep it cheap

MODELS = [
    "anthropic/claude-opus-4.8",
    "openai/gpt-5.6-terra",
    "google/gemini-3.5-flash",
    "deepseek/deepseek-chat",
]

TASKS = [
    ("rust", "an LRU cache with get/put"),
    ("go", "a token-bucket rate limiter"),
    ("typescript", "a debounce function"),
    ("c", "a fixed-capacity ring buffer"),
    ("zig", "an iterative quicksort over a slice"),
    ("python", "a retry-with-exponential-backoff decorator"),
    ("javascript", "a trie with insert and prefix search"),
    ("rust", "a thread-safe counter using atomics"),
    ("c", "a singly linked list with insert and delete"),
    ("python", "a bloom filter"),
    ("go", "an iterative merge sort of an int slice"),
    ("typescript", "a small typed event emitter"),
    ("zig", "a UTF-8 byte-sequence decoder"),
    ("javascript", "a binary min-heap"),
    ("c", "a simple JSON string tokenizer"),
    ("python", "a memoization cache with TTL"),
    ("rust", "a bounded SPSC queue"),
    ("go", "a worker pool with a concurrency limit"),
    ("javascript", "a promise pool limiting concurrency"),
    ("c", "string interning with a hash table"),
    ("python", "an RFC-ish CSV row parser"),
    ("rust", "a simple binary search tree with insert"),
    ("zig", "a fixed-size stack"),
    ("go", "a context-aware timeout wrapper"),
    # --- OPEN-ENDED / design-heavy tasks: room for a model's style to show ---
    ("rust", "a small in-memory key-value store with TTL expiry — your own design"),
    ("python", "a minimal templating engine with {{var}} substitution and loops"),
    ("typescript", "a tiny reactive signals library (signal/computed/effect)"),
    ("c", "a mini stack-based arithmetic bytecode VM"),
    ("javascript", "a small finite-state-machine library with guards"),
    ("go", "a minimal in-memory pub/sub message bus"),
    ("rust", "a tiny parser-combinator library"),
    ("python", "a small dependency-injection container"),
    ("zig", "a simple arena allocator"),
    ("javascript", "a minimal virtual-DOM diff and patch"),
    ("go", "a generic LRU with per-entry expiry"),
    ("typescript", "a Result/Either error type with map/andThen helpers"),
    ("python", "a small event-sourcing aggregate with apply/replay"),
    ("rust", "a simple thread pool with a shared work queue"),
    ("javascript", "a tiny observable/stream with map and filter"),
    ("c", "a small slab allocator for fixed-size objects"),
]


def call(model, lang, task):
    prompt = (f"Write {lang} code implementing {task}. Output ONLY the code — no prose, no "
              f"comments beyond what you'd normally write, no markdown fences.")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.7,
        "reasoning": {"effort": "low"},  # keep reasoning models from eating the budget
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]["content"] or ""
    cost = float(d.get("usage", {}).get("cost", 0.0) or 0.0)
    return msg, cost


def strip_fences(s):
    s = s.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    return s.strip()


n_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else len(TASKS)
tasks = TASKS[:n_tasks]
# RESUMABLE: load prior samples, skip (model,task) already done, write after EACH call.
corpus = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {(c["model"], c["task"]) for c in corpus}
total = 0.0
print(f"generating {len(tasks)} tasks x {len(MODELS)} models; {len(done)} already done; cap ${COST_CAP}")
for model in MODELS:
    for lang, task in tasks:
        if (model, task) in done:
            continue
        if total >= COST_CAP:
            print(f"!! COST CAP ${COST_CAP} hit — stopping"); break
        try:
            code, cost = call(model, lang, task)
            code = strip_fences(code)
            total += cost
            corpus.append({"model": model, "lang": lang, "task": task, "code": code,
                           "chars": len(code), "cost": cost})
            done.add((model, task))
            json.dump(corpus, open(OUT, "w"), indent=1)  # write immediately (kill-safe)
            print(f"  {model:<30} {lang:<11} {task[:26]:<26} {len(code):>4}ch ${cost:.4f}  run=${total:.4f}")
        except Exception as e:
            print(f"  {model:<30} {lang:<11} FAILED: {str(e)[:60]}")
        time.sleep(0.3)
    if total >= COST_CAP:
        break
print(f"\nwrote {len(corpus)} samples -> {OUT}")
print(f"TOTAL SPENT THIS RUN: ${total:.4f}")
by = {}
for c in corpus:
    by.setdefault(c["model"], []).append(c["chars"])
for m, cs in by.items():
    print(f"  {m:<30} {len(cs)} samples, mean {sum(cs)//len(cs)} chars")
