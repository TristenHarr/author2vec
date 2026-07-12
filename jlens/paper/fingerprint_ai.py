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
# model centroids built from all OTHER tasks). `assign` maps (model,task)->label so we can
# recompute the exact statistic under a permutation null.
cells = list(E.keys())  # (model, task) cells that actually embedded


def ctrl_accuracy(assign):
    correct = tot = 0
    for (m, t) in cells:
        # build a centroid per assigned label from all cells NOT in task t
        acc_sum, acc_cnt = {}, {}
        for (x, tt) in cells:
            if tt == t:
                continue
            lab = assign[(x, tt)]
            if lab in acc_sum:
                acc_sum[lab] = acc_sum[lab] + E[(x, tt)]
                acc_cnt[lab] += 1
            else:
                acc_sum[lab] = E[(x, tt)].copy()
                acc_cnt[lab] = 1
        cents = {lab: unit(acc_sum[lab] / acc_cnt[lab]) for lab in acc_sum}
        pred = max(cents, key=lambda L: cos(E[(m, t)], cents[L]))
        correct += pred == assign[(m, t)]
        tot += 1
    return correct / tot


true_assign = {k: k[0] for k in E}
ctrl_acc = ctrl_accuracy(true_assign)

# Permutation null: within each task, shuffle which model each sample is attributed to. This
# keeps the task x model grid intact but breaks the true model->style association. p = fraction
# of relabelings that match or beat the observed accuracy.
rng = np.random.default_rng(0)
N_PERM = 1000
null = []
by_task = {t: [m for m in models if (m, t) in E] for t in tasks}
for _ in range(N_PERM):
    perm = {}
    for t in tasks:
        ms = by_task[t]
        shuf = list(ms)
        rng.shuffle(shuf)
        for m, lab in zip(ms, shuf):
            perm[(m, t)] = lab
    null.append(ctrl_accuracy(perm))
null = np.array(null)
p_val = float((np.sum(null >= ctrl_acc) + 1) / (N_PERM + 1))

# grid accounting: 40 tasks x 4 models = 160 cells; empty generations drop out
n_grid = len(models) * len(tasks)
n_valid = len(E)

print("=== AI MODEL FINGERPRINT (task-controlled) ===")
print(f"{len(E)}/{n_grid} valid cells, {len(models)} models, {len(tasks)} tasks")
print(f"task effect  (same task, diff model): {np.mean(task_eff):.3f}")
print(f"model effect (same model, diff task): {np.mean(model_eff):.3f}")
print(f"-> {'TASK dominates' if np.mean(task_eff) > np.mean(model_eff) else 'MODEL dominates'}")
print(f"task-controlled model ID: {ctrl_acc*100:.1f}%  (chance {100/len(models):.0f}%), "
      f"null mean {null.mean()*100:.1f}%, p={p_val:.4f}")

out = {
    "n_samples": n_valid,
    "n_grid": n_grid,
    "n_tasks": len(tasks),
    "n_tasks_canonical": 24,
    "n_tasks_open_ended": 16,
    "models": [short[m] for m in models],
    "task_effect": round(float(np.mean(task_eff)), 3),
    "model_effect": round(float(np.mean(model_eff)), 3),
    "task_controlled_model_id": round(ctrl_acc, 3),
    "task_controlled_null_mean": round(float(null.mean()), 3),
    "task_controlled_p": round(p_val, 4),
    "chance": round(1 / len(models), 3),
    "verdict": "task dominates; faint but real model fingerprint" if p_val < 0.05
    else "no model fingerprint above chance",
}
json.dump(out, open(os.path.join(ASSETS, "person2vec-aifp.json"), "w"), indent=1)
print("wrote person2vec-aifp.json")
