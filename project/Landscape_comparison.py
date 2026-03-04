import os, json, random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import esm
from scipy.stats import mannwhitneyu, ks_2samp, skew
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CFG = dict(
    train_csv="project_data/algpred2_train_seq.csv",
    test_csv="project_data/algpred2_test_seq.csv",
    ckpt_path="ddg_best_20260227_142913.ptrom",
    out_dir="landscape_output",
    max_positions_per_protein=50,   # sample 50 positions → ~950 mutants/protein (set None for full)
    amino_acids="ACDEFGHIKLMNPQRSTVWY",
    batch_size_mut=128,
    thr_destab=1.0,
    thr_neutral=0.5,
    label_allergen=1,
    label_non_allergen=0,
)

os.makedirs(CFG["out_dir"], exist_ok=True)
out_path = lambda name: os.path.join(CFG["out_dir"], name)
LOG_PATH = out_path("run_log.txt")

def log(msg):
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


# ── Model ─────────────────────────────────────────────────────────────────────

class DDGPredictor(nn.Module):
    def __init__(self, esm_model, alphabet, hidden=256, dropout=0.2):
        super().__init__()
        self.esm, self.alphabet = esm_model, alphabet
        self.repr_layer = esm_model.num_layers
        d = esm_model.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(4*d), nn.Linear(4*d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def encode(self, tokens):
        h = self.esm(tokens, repr_layers=[self.repr_layer], return_contacts=False)["representations"][self.repr_layer]
        mask = (tokens != self.alphabet.padding_idx) & (tokens != self.alphabet.eos_idx)
        mask[:, 0] = False
        return (h * mask.unsqueeze(-1).float()).sum(1) / mask.sum(1, keepdim=True).clamp_min(1).float()

    def forward(self, wt, mut):
        w, m = self.encode(wt), self.encode(mut)
        return self.head(torch.cat([w, m, m - w, m * w], dim=-1)).squeeze(-1)


def load_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: '{ckpt_path}' (cwd: {os.getcwd()})")
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    for p in esm_model.parameters():
        p.requires_grad = False
    model = DDGPredictor(esm_model, alphabet).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    log("[warn] Missing keys: "    + ", ".join(missing[:20]))
    if unexpected: log("[warn] Unexpected keys: " + ", ".join(unexpected[:20]))
    model.eval(); model.esm.eval()
    return model, alphabet


# ── Mutation generation ────────────────────────────────────────────────────────

AA20 = list(CFG["amino_acids"])

def sample_positions(L, k):
    return list(range(L)) if k is None or k >= L else sorted(np.random.choice(L, k, replace=False).tolist())

def generate_single_mutants(seq, max_positions=None):
    seq = list(seq.strip().upper())
    for i in sample_positions(len(seq), max_positions):
        if seq[i] not in AA20:
            continue
        orig = seq[i]
        for mut in AA20:
            if mut == orig:
                continue
            seq[i] = mut
            yield "".join(seq)
        seq[i] = orig


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_ddg(model, batch_converter, wt_id, wt_tok_single, mutant_seqs):
    """wt_tok_single: [1, L] tensor tokenised once upstream — expanded here per batch."""
    n = len(mutant_seqs)
    wt_tok = wt_tok_single.expand(n, -1).to(DEVICE)           # reuse, no re-tokenise
    _, _, mut_tok = batch_converter(list(zip([wt_id] * n, mutant_seqs)))
    return model(wt_tok, mut_tok.to(DEVICE)).cpu().numpy()


# ── Landscape descriptors ──────────────────────────────────────────────────────

def landscape_features(ddg_vals):
    v = np.asarray(ddg_vals, dtype=float)
    v = v[~np.isnan(v)]
    nan = float("nan")
    if v.size == 0:
        return dict(n_mut=0, mean=nan, median=nan, std=nan, skew=nan,
                    frac_destab=nan, frac_strong_destab=nan, frac_stab=nan, frac_neutral=nan,
                    q10=nan, q25=nan, q75=nan, q90=nan)
    td, tn = CFG["thr_destab"], CFG["thr_neutral"]
    return dict(
        n_mut=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
        std=float(v.std()), skew=float(skew(v)) if v.size >= 3 else nan,
        frac_destab=float((v > td).mean()), frac_strong_destab=float((v > 2.0).mean()),
        frac_stab=float((v < -1.0).mean()), frac_neutral=float((np.abs(v) < tn).mean()),
        q10=float(np.quantile(v, 0.10)), q25=float(np.quantile(v, 0.25)),
        q75=float(np.quantile(v, 0.75)), q90=float(np.quantile(v, 0.90)),
    )


# ── IO ─────────────────────────────────────────────────────────────────────────

def load_sequences(csv_path):
    df = pd.read_csv(csv_path)
    for c in ["id", "sequence", "label"]:
        if c not in df.columns:
            raise ValueError(f"{csv_path} missing column '{c}'")
    df = df[["id", "sequence", "label"]].copy()
    df["sequence"] = df["sequence"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    return df


# ── Compute landscapes ─────────────────────────────────────────────────────────

def compute_landscapes(df, model, alphabet):
    batch_converter = alphabet.get_batch_converter()   # created once
    rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Proteins"):
        pid   = str(r["id"])
        seq   = str(r["sequence"]).strip().upper()
        label = int(r["label"]) if pd.notna(r["label"]) else None

        # Tokenise WT once per protein
        _, _, wt_tok_single = batch_converter([(pid, seq)])

        ddg_all, batch = [], []
        for mut_seq in generate_single_mutants(seq, CFG["max_positions_per_protein"]):
            batch.append(mut_seq)
            if len(batch) >= CFG["batch_size_mut"]:
                ddg_all.append(predict_ddg(model, batch_converter, pid, wt_tok_single, batch))
                batch = []
        if batch:
            ddg_all.append(predict_ddg(model, batch_converter, pid, wt_tok_single, batch))

        ddg_vals = np.concatenate(ddg_all) if ddg_all else np.array([], dtype=float)
        rows.append(dict(
            id=pid, label=label, length=len(seq),
            positions_used=len(sample_positions(len(seq), CFG["max_positions_per_protein"])),
            **landscape_features(ddg_vals),
        ))

    return pd.DataFrame(rows)


# ── Group comparison ───────────────────────────────────────────────────────────

def compare_groups(per_protein, feature_cols):
    gA = per_protein[per_protein["label"] == CFG["label_allergen"]]
    gB = per_protein[per_protein["label"] == CFG["label_non_allergen"]]
    lines = [f"n_allergen: {len(gA)}", f"n_non_allergen: {len(gB)}", ""]
    for col in feature_cols:
        A, B = gA[col].dropna().values, gB[col].dropna().values
        if len(A) < 3 or len(B) < 3:
            lines.append(f"[{col}] not enough data"); continue
        u  = mannwhitneyu(A, B, alternative="two-sided")
        ks = ks_2samp(A, B)
        lines.append(
            f"[{col}] median(allergen)={np.median(A):.4f}, median(non)={np.median(B):.4f} "
            f"| MWU p={u.pvalue:.3e} | KS p={ks.pvalue:.3e}"
        )
    p = out_path("ddg_landscape_group_stats.txt")
    open(p, "w").write("\n".join(lines))
    log(f"Saved group stats: {p}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_feature(per_protein, col, filename):
    A = per_protein[per_protein["label"] == CFG["label_allergen"]][col].dropna().values
    B = per_protein[per_protein["label"] == CFG["label_non_allergen"]][col].dropna().values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([B, A], labels=["non-allergen", "allergen"], showfliers=False)
    ax.set_ylabel(col); ax.set_title(f"Protein-level {col}")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p = out_path(filename)
    plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    log(f"Saved plot: {p}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log(f"=== Run started {datetime.now().isoformat(timespec='seconds')} | Device: {DEVICE} ===")
    with open(out_path("config_used.json"), "w") as f:
        json.dump(CFG, f, indent=2)

    model, alphabet = load_model(CFG["ckpt_path"])
    log("Loaded model.")

    df = pd.concat([
        load_sequences(CFG["train_csv"]).assign(split="train"),
        load_sequences(CFG["test_csv"]).assign(split="test"),
    ], ignore_index=True)
    log("Label distribution: " + str(df["label"].value_counts(dropna=False).to_dict()))

    per_protein = compute_landscapes(df, model, alphabet)
    p = out_path("ddg_landscape_per_protein.csv")
    per_protein.to_csv(p, index=False)
    log(f"Saved per-protein features: {p}")

    feature_cols = ["mean", "median", "std", "skew", "frac_destab", "frac_strong_destab",
                    "frac_stab", "frac_neutral", "q10", "q90"]
    compare_groups(per_protein, feature_cols)
    plot_feature(per_protein, "frac_destab", "ddg_landscape_frac_destab.png")
    plot_feature(per_protein, "mean", "ddg_landscape_mean.png")
    log(f"=== Run finished {datetime.now().isoformat(timespec='seconds')} ===")


if __name__ == "__main__":
    main()