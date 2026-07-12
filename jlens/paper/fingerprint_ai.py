#!/usr/bin/env python3
"""AI-model fingerprint — the HONEST, task-controlled version.

Naive cross-model similarity is dominated by the task (same task from different models looks
near-identical), so it CANNOT support a 'models converged' claim. We instead report:
  (1) task effect  = mean cos(same task, different model)
  (2) model effect = mean cos(same model, different task)
  (3) task-controlled model ID = leave-one-task-out nearest-model-centroid accuracy
A model fingerprint exists iff (3) beats chance. Ground truth: we generated the code.

Inputs: scratchpad/ai_corpus.json + ai_embed_out.json. Writes person2vec-aifp.json + prints.
"""
import json
import os
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-tristenharr-duhclaudeknowsyou/4d4c9584-9d59-4b78-a64a-36a229af2782/scratchpad"
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ASSETS = os.path.join(ROOT, "web", "assets")


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-9)


emb = {e["id"]: unit(e["vec"]) for e in json.load(open(os.path.join(SCRATCH, "ai_embed_out.json")))}
corp = json.load(open(os.path.join(SCRATCH, "ai_corpus.json")))
E = {}  # (model, task) -> unit vec
for i, x in enumerate(corp):
    k = f"{x['model']}##{i}"
    if k in emb:
        E[(x["model"], x["task"])] = emb[k]
models = sorted({m for m, _ in E})
tasks = sorted({t for _, t in E})
short = {m: m.split("/")[-1] for m in models}


def cos(a, b):
    return float(a @ b)


task_eff = [cos(E[(m, t)], E[(m2, t)]) for t in tasks for i, m in enumerate(models)
            for m2 in models[i + 1:] if (m, t) in E and (m2, t) in E]
model_eff = [cos(E[(m, t)], E[(m, t2)]) for m in models for i, t in enumerate(tasks)
             for t2 in tasks[i + 1:] if (m, t) in E and (m, t2) in E]

# leave-one-task-out model ID (controls task: held-out task's samples classified by
# model centroids built from all OTHER tasks).
correct = tot = 0
for t in tasks:
    for m in models:
        if (m, t) not in E:
            continue
        cents = {}
        for mm in models:
            others = [E[(mm, tt)] for tt in tasks if tt != t and (mm, tt) in E]
            if others:
                cents[mm] = unit(np.mean(others, 0))
        pred = max(cents, key=lambda mm: cos(E[(m, t)], cents[mm]))
        correct += pred == m
        tot += 1
ctrl_acc = correct / tot

print("=== AI MODEL FINGERPRINT (task-controlled) ===")
print(f"{len(E)} samples, {len(models)} models, {len(tasks)} tasks")
print(f"task effect  (same task, diff model): {np.mean(task_eff):.3f}")
print(f"model effect (same model, diff task): {np.mean(model_eff):.3f}")
print(f"-> {'TASK dominates' if np.mean(task_eff) > np.mean(model_eff) else 'MODEL dominates'}")
print(f"task-controlled model ID: {ctrl_acc*100:.1f}%  (chance {100/len(models):.0f}%)")

out = {
    "n_samples": len(E),
    "n_tasks": len(tasks),
    "models": [short[m] for m in models],
    "task_effect": round(float(np.mean(task_eff)), 3),
    "model_effect": round(float(np.mean(model_eff)), 3),
    "task_controlled_model_id": round(ctrl_acc, 3),
    "chance": round(1 / len(models), 3),
    "verdict": "task dominates; faint but real model fingerprint" if ctrl_acc > 1.2 / len(models)
    else "no model fingerprint above chance",
}
json.dump(out, open(os.path.join(ASSETS, "person2vec-aifp.json"), "w"), indent=1)
print("wrote person2vec-aifp.json")
