import os, json, random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, root_mean_squared_error
from transformers import pipeline # IMPORTANT: Please ensure your transformers version is v4.55
import evaluate

# ---------------- Config ----------------
GOLD_FILE = "../data/tsar2025_test.jsonl"   # gold file next to this script
SUBMISSIONS_DIR = "../submissions"     # folder with team subfolders
SEED = 42                           # for reproducibility
BATCH_SIZE = 32                     # adjust for your GPU

# ---------------- Seed ------------------
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

# ---------------- IO --------------------
def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            line = line.strip() # Remove leading and trailing spaces and newlines
            if line: # The line will only be parsed if it is not empty.
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data

def read_gold(path: str):
    data = read_jsonl(path)
    if not data:
        raise ValueError(f"Gold file is empty: {path}")
    try:
        original = [e["original"] for e in data]
        reference = [e["reference"] for e in data]
        target   = [e["target_cefr"] for e in data]   # case handled later
        text_ids = [e["text_id"] for e in data]
    except KeyError as ke:
        raise KeyError(f"Gold file missing key {ke}. First item keys: {list(data[0].keys())}")
    return original, reference, target, text_ids

def read_submission(path: str):
    data = read_jsonl(path)
    if not data:
        raise ValueError(f"Submission is empty: {path}")
    first_keys = list(data[0].keys())
    if "simplified" not in data[0]:
        raise KeyError(f"{path} must contain 'simplified'. Found keys: {first_keys}")
    if "text_id" not in data[0]:
        raise KeyError(f"{path} must contain 'text_id'. Found keys: {first_keys}")
    return [e["simplified"] for e in data], [e["text_id"] for e in data], len(data)

# Align system outputs to ANY overlapping gold ids (supports partial submissions)
def align_intersection(hyps, sys_ids, gold_ids, gold_orig, gold_ref, gold_tgt):
    gid2idx = {g:i for i,g in enumerate(gold_ids)}
    pairs = [(gid2idx[sid], hyp) for hyp, sid in zip(hyps, sys_ids) if sid in gid2idx]
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    sel_idx = [i for i,_ in pairs]
    aligned_hyps = [h for _,h in pairs]
    aligned_orig = [gold_orig[i] for i in sel_idx]
    aligned_ref  = [gold_ref[i]  for i in sel_idx]
    aligned_tgt  = [gold_tgt[i]  for i in sel_idx]
    coverage_n   = len(sel_idx)
    coverage_pct = round(100.0 * coverage_n / len(gold_ids), 2)
    missing_ids  = [g for g in gold_ids if g not in set(sys_ids)]
    extra_ids    = [s for s in sys_ids if s not in set(gold_ids)]
    return {
        "hyps": aligned_hyps,
        "orig": aligned_orig,
        "ref":  aligned_ref,
        "tgt":  aligned_tgt,
        "coverage_n": coverage_n,
        "coverage_pct": coverage_pct,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids
    }

# ------------- Models/Metrics -----------
cefr_labeler1 = pipeline("text-classification",
    model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr", device=0, torch_dtype="auto")
cefr_labeler2 = pipeline("text-classification",
    model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr", device=0, torch_dtype="auto")
cefr_labeler3 = pipeline("text-classification",
    model="AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2", device=0, torch_dtype="auto")

meaning_bert = evaluate.load("davebulaval/meaningbert")
bertscore    = evaluate.load("bertscore")

CEFR = ["A1","A2","B1","B2","C1","C2"]
L2I  = {l:i for i,l in enumerate(CEFR)}

def cefr_labels(hyps, models, batch_size=BATCH_SIZE):
    p1 = models[0](hyps, batch_size=batch_size, truncation=True)
    p2 = models[1](hyps, batch_size=batch_size, truncation=True)
    p3 = models[2](hyps, batch_size=batch_size, truncation=True)
    def top1(x):
        if isinstance(x, dict): return x
        if isinstance(x, list) and x: return max(x, key=lambda d: d["score"])
    outs = []
    for d1, d2, d3 in zip(p1, p2, p3):
        best = max((top1(d1), top1(d2), top1(d3)), key=lambda d: d["score"])
        outs.append(best["label"].strip().upper())
    return outs

def score_cefr(hyps, ref_lvls, models):
    gold  = [str(l).strip().upper() for l in ref_lvls]
    preds = [str(l).strip().upper() for l in cefr_labels(hyps, models, batch_size=BATCH_SIZE)]
    f1 = f1_score(gold, preds, average="weighted")
    t  = np.array([L2I[l] for l in gold])
    p  = np.array([L2I[l] for l in preds])
    adj  = (np.abs(t - p) <= 1).mean()
    rmse = root_mean_squared_error(t, p)
    return {"weighted_f1": round(float(f1),4),
            "adj_accuracy": round(float(adj),4),
            "rmse": round(float(rmse),4)}

def score_meaningbert(hyps, refs):
    res = meaning_bert.compute(predictions=hyps, references=refs)
    return round(float(np.mean(res["scores"])) / 100.0, 4)

def score_bertscore(hyps, refs, scoretype="f1"):
    res = bertscore.compute(references=refs, predictions=hyps, lang="en")
    return round(float(np.mean(res[scoretype])), 4)

# ------------- Main ---------------------
if not os.path.isfile(GOLD_FILE):
    raise FileNotFoundError(f"Gold file not found: {GOLD_FILE}")
gold_orig, gold_ref, gold_tgt, gold_ids = read_gold(GOLD_FILE)

if not os.path.isdir(SUBMISSIONS_DIR):
    raise FileNotFoundError(f"Submissions folder not found: {SUBMISSIONS_DIR}")

team_dirs = sorted([d for d in os.listdir(SUBMISSIONS_DIR)
                    if os.path.isdir(os.path.join(SUBMISSIONS_DIR, d)) and not d.startswith(".")])

results = []
for team in team_dirs:
    team_path = os.path.join(SUBMISSIONS_DIR, team)
    run_files = sorted([f for f in os.listdir(team_path) if f.endswith(".jsonl")])
    if not run_files:
        print(f"[warn] No .jsonl files in {team_path}")
        continue
    for run in run_files:
        run_path = os.path.join(team_path, run)
        print(f"Evaluating {team}/{run} ...")
        hyps, sys_ids, num_instances = read_submission(run_path)

        aligned = align_intersection(hyps, sys_ids, gold_ids, gold_orig, gold_ref, gold_tgt)
        if aligned is None:
            print(f"[{team}/{run}] no overlap with gold; skipping.")
            row = {"modelname": run, "teamname": team,
                   "num_instances": num_instances,
                   "coverage_n": 0, "coverage_pct": 0.0,
                   "weighted_f1": "n/a", "adj_accuracy": "n/a", "rmse": "n/a",
                   "meaningbert-orig": "n/a", "bertscore-orig": "n/a",
                   "meaningbert-ref": "n/a", "bertscore-ref": "n/a"}
        else:
            if aligned["missing_ids"]:
                print(f"[{team}/{run}] missing {len(aligned['missing_ids'])} ids.")
            if aligned["extra_ids"]:
                print(f"[{team}/{run}] extra {len(aligned['extra_ids'])} ids (ignored).")
            hyps_i, orig_i, ref_i, tgt_i = aligned["hyps"], aligned["orig"], aligned["ref"], aligned["tgt"]
            cefr = score_cefr(hyps_i, tgt_i, [cefr_labeler1, cefr_labeler2, cefr_labeler3])
            mb_o = score_meaningbert(hyps_i, orig_i)
            bs_o = score_bertscore(hyps_i, orig_i, "f1")
            mb_r = score_meaningbert(hyps_i, ref_i)
            bs_r = score_bertscore(hyps_i, ref_i, "f1")
            row = {"modelname": run, "teamname": team,
                   "num_instances": num_instances,
                   "coverage_n": aligned["coverage_n"],
                   "coverage_pct": aligned["coverage_pct"],
                   "weighted_f1": cefr["weighted_f1"], "adj_accuracy": cefr["adj_accuracy"], "rmse": cefr["rmse"],
                   "meaningbert-orig": mb_o, "bertscore-orig": bs_o,
                   "meaningbert-ref": mb_r, "bertscore-ref": bs_r}
        results.append(row)

df = pd.DataFrame(results)
print("\n=== Results ===")
print(df.to_string(index=False))
df.to_excel("results.xlsx", index=False)
print("\nSaved: results.xlsx")
