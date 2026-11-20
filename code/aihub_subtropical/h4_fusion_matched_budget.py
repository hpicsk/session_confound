"""
h4_fusion_matched_budget.py — the fusion experiment at the CDD study's training budget.

WHAT THIS MEASURES. At the reduced budget (h4_fusion_collapse.py, 40 epochs) the in-distribution
fusion gain has an interval that includes zero: delta_iid = +0.0013 [-0.0058, +0.0084]. 40
epochs may undertrain the CDD study's environmental branch (each image is seen 40 times here,
against 300 in the CDD study). This script replicates the CDD study's training budget on the AI Hub
dataset — 300 epochs, batch 8, NO early stopping, best-validation checkpoint — on the
in-distribution split only.

THE OUTCOME: the run returned an in-distribution gain whose interval excludes zero
(sec:fusion), because the image-only model lost more macro-F1 at the matched budget than the
fusion model did. The in-distribution leak rate is near-total (89.8% of test images carry a
panel tuple that recurs in training, h4_positivity_recheck.csv).

The printed outcome is the same as at the reduced budget: whether the in-distribution delta's
CI excludes 0. The manuscript reports the gain and its interval (sec:fusion), and that this
training was added after the reduced-budget run showed no gain (sec:fusion-design). The reduced-budget table h4_fusion_collapse.csv is NOT
touched; this script writes h4_fusion_matched_budget.csv.

ORDER OF THE RUNS:
  - in-distribution only, 2 models x 5 seeds = 10 trainings, run FIRST.
  - if the gain's CI includes 0, the epoch-deficiency objection is closed on the AI Hub dataset
    and no split that holds out dates or farms is run.
  - if the gain's CI excludes 0, the date-disjoint run follows (h4_g4b_collapse.py).

DEGREES OF FREEDOM: identical to the reduced-budget run — MEASURED: only the per-split min-max
bounds (TRAIN rows). FITTED: none (no knob tuned to a macro-F1 target). CHOSEN: as in
h4_fusion_collapse.py, with the three budget constants below re-set to the CDD study's values.
BASELINE: ResNet-50 + the CDD study's late-fusion sensor branch + its 300-epoch/batch-8
  training procedure, held identical across the two models | obvious alternative a referee
  names: "a stronger fusion head would produce a larger gain" | tuning: none searched --
  backbone, branch and training procedure are transplanted from the CDD study, and the three
  budget constants are its published values | discharge: sec:discussion, which claims no upper
  bound on a fusion design we did not run.

INHERITED VERBATIM from h4_fusion_collapse.py (the reduced-budget script, kept as committed —
imported, not copied, so a single source stays authoritative): DaconLSTM/DaconModel (= the CDD
study's src/model.py), build_model, train_tf/eval_tf augmentation, PanelDS, load_metadata, the
in-distribution carve, per-split TRAIN-only normalization, evaluate (best-val-checkpoint
macro-F1), ci, load_images, seed_everything, and the LR/BETAS/T_0/N_CLASSES/subsample constants.

CHANGED for the matched budget (the ONLY differences from the reduced-budget script; each cited
to the CDD study's repository):
  epochs      40 -> 300              src/main.py:24
  batch       64 -> 8               script/train_single_modal.sh:3
  early-stop  patience-5 -> NONE    the CDD study's train loop has no break -> hard 300
                                    epochs, best-val checkpoint kept
  splits      {iid,date} -> {iid}   date run by h4_g4b_collapse.py
  output      h4_fusion_collapse.csv -> h4_fusion_matched_budget.csv
The loader and train loop are re-stated below (not imported) precisely BECAUSE batch and epochs
are module globals in h4_fusion_collapse.py; re-stating them here keeps that script unchanged
while making every changed constant explicit in one place (a config bug here costs ~3 GPU-days).

Run (each mode explicit; bare invocation prints usage and trains nothing):
  TROPICAL_SCRATCH=/path/to/extraction TROPICAL_GPU=0 python h4_fusion_matched_budget.py time  # ~3min timing run
  TROPICAL_SCRATCH=/path/to/extraction TROPICAL_GPU=0 python h4_fusion_matched_budget.py run   # the GPU run
"""

import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# Single source: h4_fusion_collapse.py owns the model, data, augmentation, metric and pure helpers.
# Importing (not copying) keeps DaconModel/loader-semantics/metric identical across the two budgets
# so the only differences are the four budget constants documented in the header.
import h4_fusion_collapse as base
from h4_fusion_collapse import (  # config-free helpers whose base globals equal ours (N_CLASSES=18, PX=256, ...)
    PanelDS, build_model, carve, ci, eval_tf, evaluate, load_images,
    load_metadata, normalize, seed_everything, train_tf,
)

# ---- matched-budget design; none tuned ----
BS = 8                       # CDD script/train_single_modal.sh:3  (reduced budget used 64)
EPOCHS = 300                 # CDD src/main.py:24                  (reduced budget used 40)
# early stopping: NONE (the CDD study runs the full ceiling; best-val checkpoint kept). The reduced budget used patience 5.
SEEDS = (0, 1, 2, 3, 4)      # the 5 seeds of the reduced budget
ARMS = ("img", "all")        # image-only model . late-fusion model (DaconModel)
SPLIT = "iid"                # in-distribution only (date: h4_g4b_collapse.py)

OUT = base.HERE / "h4_fusion_matched_budget.csv"     # its own file; the reduced-budget table stays as committed


def loader(imgs, env, y, idx, tf, arm, train, seed):
    """The CDD study's inverse-frequency WeightedRandomSampler on train, seeded from `seed` alone so
    both models draw the identical order within a seed (pairs the delta). Re-stated from
    h4_fusion_collapse.py with BS=8 — the only change is the batch size (see header)."""
    ds = PanelDS(imgs, env, y, idx, tf, arm != "csv", arm == "all")
    if not train:
        return DataLoader(ds, batch_size=BS, shuffle=False, num_workers=8, pin_memory=True)
    classes, counts = np.unique(y[idx], return_counts=True)
    cw = counts.sum() / counts
    remap = {c: k for k, c in enumerate(classes)}  # robust if a fold omits a class
    w = np.array([cw[remap[int(y[j])]] for j in idx], dtype=np.float64)
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.DoubleTensor(w), len(idx), replacement=True, generator=g)
    return DataLoader(ds, batch_size=BS, sampler=sampler, num_workers=8, pin_memory=True)


def train_arm(imgs, env_raw, y, tr, va, te, seed, arm):
    """One faithful training: Adam + CosineAnnealingWarmRestarts, best-val-macro-F1 checkpoint,
    HARD 300 epochs with NO early stopping (the CDD study's loop shape). Re-stated from
    h4_fusion_collapse.py with EPOCHS=300 and the patience break removed
    — those are the only changes. Per-epoch progress is printed for the timing/monitor only and
    does not touch training."""
    seed_everything(seed)
    env = normalize(env_raw, tr) if arm == "all" else env_raw  # the image-only model ignores env
    model = build_model(arm)
    opt = torch.optim.Adam(model.parameters(), lr=base.LR, betas=base.BETAS)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=base.T_0, eta_min=base.LR / 1e3)
    lossf = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=base.DEV.startswith("cuda"))

    dl_tr = loader(imgs, env, y, tr, train_tf(), arm, True, seed)
    dl_va = loader(imgs, env, y, va, eval_tf(), arm, False, seed)

    t0 = time.time()
    best_val, best_state, best_ep = -1.0, None, -1
    for ep in range(EPOCHS):                       # HARD 300 — no early stop (matched to the CDD study)
        model.train()
        for img, e, yb in dl_tr:
            yb = yb.to(base.DEV, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=base.DEV.startswith("cuda")):
                out = model(img.to(base.DEV), e.to(base.DEV)) if arm == "all" else model(img.to(base.DEV))
                loss = lossf(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        vf1, _ = evaluate(model, dl_va, arm)
        if vf1 > best_val:
            best_val, best_ep, best_state = vf1, ep, copy.deepcopy(model.state_dict())
        if ep < 3 or (ep + 1) % 25 == 0 or ep == EPOCHS - 1:  # progress only; no training effect
            print(f"      ep {ep + 1:3d}/{EPOCHS} val_f1={vf1:.4f} best={best_val:.4f}@{best_ep} "
                  f"{(time.time() - t0) / 60:.1f}min", flush=True)

    model.load_state_dict(best_state)
    tf1, tba = evaluate(model, loader(imgs, env, y, te, eval_tf(), arm, False, seed), arm)
    return dict(test_f1=tf1, test_ba=tba, val_f1=best_val, best_epoch=best_ep,
                n_train=len(tr), n_val=len(va), n_test=len(te),
                n_class_train=int(len(np.unique(y[tr]))), n_class_test=int(len(np.unique(y[te]))),
                wall_s=time.time() - t0)


def report(long):
    """The printed outcome at the matched budget, the same as at the reduced budget: whether
    the in-distribution delta's CI excludes 0."""
    w = long.pivot_table(index=["split", "seed"], columns="arm", values="test_f1").reset_index()
    w["delta"] = w["all"] - w["img"]
    g = w[w.split == SPLIT]
    md, ld, hd = base.ci(g["delta"].values)
    print("\n" + "=" * 88)
    print("MATCHED BUDGET (300 epochs / batch 8, no early stop) — in-distribution delta = all - img, 5 seeds paired")
    print("=" * 88)
    print(f"  iid  delta={md:+.4f} [{ld:+.4f}, {hd:+.4f}]   "
          f"img={g['img'].mean():.4f} all={g['all'].mean():.4f}   n_seeds={len(g)}")
    replicates = ld > 0
    print(f"\n  replicates (in-distribution delta CI excludes 0): {replicates}")
    if replicates:
        print("  The gain replicates at 300 epochs although it did not at 40. The\n"
              "  date-disjoint run follows (h4_g4b_collapse.py).")
    else:
        print("  The gain does not replicate at 300 epochs either, even at the CDD study's\n"
              "  budget. The epoch-deficiency objection is closed on the AI Hub dataset; no split\n"
              "  that holds out dates or farms is run.")
    return dict(replicates=replicates, d=md, lo=ld, hi=hd)


def run():
    print("=" * 88)
    print(f"MATCHED-BUDGET RUN — in-distribution only, {len(ARMS)} models x {len(SEEDS)} seeds "
          f"({EPOCHS}ep / batch{BS}, no early-stop). Device {base.DEV}.")
    print("=" * 88)
    sub, y, env_raw = load_metadata()
    imgs = load_images(sub)  # fail-loud if the image cache is missing (see h4_build_image_cache.py)
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["split"], r["seed"], r["arm"]) for r in rows}
    if done:
        print(f"resuming: {len(done)}/{len(SEEDS) * len(ARMS)} trainings already in {OUT}", flush=True)
    tr, va, te = carve(sub, SPLIT)
    print(f"iid carve: tr/va/te = {len(tr)}/{len(va)}/{len(te)}  "
          f"classes tr/te = {len(np.unique(y[tr]))}/{len(np.unique(y[te]))}", flush=True)
    for seed in SEEDS:
        for arm in ARMS:
            if (SPLIT, seed, arm) in done:
                continue
            print(f"  --- iid seed={seed} {arm} start ({time.strftime('%H:%M:%S')}) ---", flush=True)
            r = train_arm(imgs, env_raw, y, tr, va, te, seed, arm)
            rows.append(dict(split=SPLIT, seed=seed, arm=arm, **r))
            print(f"  iid seed={seed} {arm:3s} f1={r['test_f1']:.4f} val={r['val_f1']:.4f} "
                  f"best_ep={r['best_epoch']} {r['wall_s'] / 60:.1f}min", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
    report(pd.DataFrame(rows))
    print(f"\nwrote {OUT}")


def time_probe():
    """~3-minute wall-clock run BEFORE the ~3-GPU-day run: train both models for 2 epochs on a
    capped subset at the real batch 8, measure sec/epoch, and extrapolate to the full 300-epoch x
    full-train run. Prints the projected total; trains no checkpoint and writes no result CSV."""
    print("=" * 88)
    print(f"TIMING PROBE (device {base.DEV}) — 2 epochs, batch{BS}, capped subset; extrapolates to "
          f"{EPOCHS}ep full-train. No result written.")
    print("=" * 88)
    sub, y, env_raw = load_metadata()
    imgs = load_images(sub)
    tr, va, te = carve(sub, SPLIT)
    cap_tr, cap_va = tr[:2400], va[:600]
    full_steps = int(np.ceil(len(tr) / BS))
    cap_steps = int(np.ceil(len(cap_tr) / BS))
    print(f"  full train {len(tr)} rows ({full_steps} steps/ep) ; probe {len(cap_tr)} rows "
          f"({cap_steps} steps/ep)", flush=True)
    for arm in ARMS:
        seed_everything(0)
        env = normalize(env_raw, tr) if arm == "all" else env_raw
        model = build_model(arm)
        opt = torch.optim.Adam(model.parameters(), lr=base.LR, betas=base.BETAS)
        lossf = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=base.DEV.startswith("cuda"))
        dl = loader(imgs, env, y, cap_tr, train_tf(), arm, True, 0)
        dl_va = loader(imgs, env, y, cap_va, eval_tf(), arm, False, 0)
        t0 = time.time()
        for ep in range(2):
            model.train()
            for img, e, yb in dl:
                yb = yb.to(base.DEV, non_blocking=True)
                opt.zero_grad()
                with torch.autocast("cuda", enabled=base.DEV.startswith("cuda")):
                    out = model(img.to(base.DEV), e.to(base.DEV)) if arm == "all" else model(img.to(base.DEV))
                    loss = lossf(out, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            evaluate(model, dl_va, arm)
        sec_ep_cap = (time.time() - t0) / 2
        sec_ep_full = sec_ep_cap * full_steps / cap_steps  # steps scale linearly with n/BS
        hrs = sec_ep_full * EPOCHS / 3600
        print(f"  {arm:3s}: {sec_ep_cap:.1f}s/ep (probe) -> ~{sec_ep_full:.1f}s/ep full "
              f"-> ~{hrs:.1f}h for {EPOCHS}ep", flush=True)
    print("\n  (multiply the per-model hours by 5 seeds, sum the two models, for the total "
          "wall-clock.)")


def main():
    ap = argparse.ArgumentParser(description="AI Hub fusion experiment at the matched budget (300 epochs, batch 8), in-distribution only")
    ap.add_argument("mode", nargs="?", choices=["time", "run"],
                    help="time: ~3min wall-clock run (trains no checkpoint). "
                         "run: the ~3-GPU-day run.")
    args = ap.parse_args()
    if args.mode == "time":
        time_probe()
    elif args.mode == "run":
        run()
    else:
        ap.print_help()
        print("\nNo mode given — nothing run. `time` probes wall-clock; `run` is the GPU run.")


if __name__ == "__main__":
    main()
