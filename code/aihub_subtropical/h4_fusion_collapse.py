"""
h4_fusion_collapse.py — the image–sensor fusion experiment on the AI Hub dataset, at the reduced
budget (40 epochs). Trains the CDD study's late-fusion model with best-validation checkpoint
selection, resized to the AI Hub dataset's 18 classes and its single nine-variable panel reading
per image.

The estimand (sec:fusion-design):

    delta(S)   = macroF1(img+env ; S) - macroF1(img ; S)      # the fusion gain under split S
    reduction  = delta(iid) - delta(date-disjoint)           # the gain lost when the leak rate falls

Backbone (ResNet-50), sensor encoder (DaconLSTM), training procedure and augmentation are
identical across the two models within a delta, so they cancel; across the two deltas the only
manipulation is the split, so `reduction` is the part of the gain lost on the date-disjoint
split, whose leak rate is far lower (sec:fusion-design). That split also changes which photo
dates and images are tested, so the reduction does not isolate the recurring panel tuples.

This script also owns the model, data, augmentation, metric and the carve that
h4_fusion_matched_budget.py and h4_g4b_collapse.py import.

------------------------------------------------------------------------------
DEGREES OF FREEDOM
------------------------------------------------------------------------------
  MEASURED  — nothing is fit to a macro-F1 target here; the only measured inputs are the
              per-split min-max normalization bounds (from TRAIN rows only, leakage-safe).
  CHOSEN (structural):
    task ...... image -> 18-way class `cls` (17 pest and disease classes and a healthy class),
                macro-F1 (= the CDD study's accuracy_function).
    models .... image-only (cnn) · late-logit fusion (DaconModel). No sensor-only model — the
                estimand needs only img and all.
    splits .... in-distribution · date-disjoint, read from h4_subsample.parquet
                (h4_materialize_splits.py fold 0; single-sourced membership).
    seeds ..... 5, {0,1,2,3,4}; mean +/- 95% CI on every number.
    val carve . fold 0 of a grouping-matched KFold/GroupKFold within TRAIN, fixed by
                SUBSAMPLE_SEED across the 5 training seeds (fixed split => seeds vary
                initialization + order only, not the split, as in the CDD study's runs).
    encoder ... DaconLSTM over the 9-col panel as a length-1 sequence; late LOGIT fusion.
                CHOSEN + BASELINE (see below).
  FITTED    — none. No hyperparameter is tuned on any fold.

  BASELINE:
    ResNet-50 + the CDD study's late-logit sensor branch + fixed training procedure, held
    identical across models and splits | obvious alternative a referee names: "a stronger
    fusion would keep the gain under the split" | discharge: sec:discussion, as a limit, not
    defended. At this file's reduced budget the in-distribution gain's interval includes zero,
    and at the matched budget (h4_fusion_matched_budget.py, h4_g4b_collapse.py) the
    date-disjoint gain and its paired reduction both include zero, so the gain did not
    detectably shrink (sec:fusion). The manuscript states that one late-fusion
    architecture sets no upper bound on what another learner could extract (sec:discussion).

------------------------------------------------------------------------------
DOCUMENTED DEVIATIONS FROM THE CDD STUDY'S CODE
------------------------------------------------------------------------------
  1. Panel is a SINGLE per-image snapshot (one 9-col row), not CDD's 320-step time series.
     It feeds the identical DaconLSTM as a length-1 sequence (B,1,9); out[:,-1,:] is that one
     step. The encoder is identical across models/splits, so this cancels in delta. This is
     inherent to the AI Hub dataset (it ships one sensor row per image), not a modeling choice.
  2. Missing panel values -> 0 before min-max (the CDD study's src/dataset.py:112
     `replace('-',0)` convention). On the AI Hub dataset this matters more than on CDD: over the
     whole spreadsheet, soil_condition is 95.7% missing, soil_K 15.9% (한라봉 has none —
     README caveat), ms_gec 0.2%. NaN->0 is deterministic, so it preserves the exact raw-tuple
     recurrence that leakage_collapse_gate.py measured (leak rate 0.90 in-distribution / 0.0735
     date); soil_condition simply degenerates to a sparse present/absent indicator, itself a
     farm signal. Some columns also carry physically impossible values: every 한라봉 soil_pH
     lies above the pH scale (up to 735), and soil_temp reaches 100. None of this
     threatens the estimand: the panel is a key from which a model recognizes a test image's
     session, not a calibrated physiological signal,
     and whatever it encodes cancels across models within a delta.
  3. Normalization bounds are fit on TRAIN rows per split (leakage-safe) rather than the CDD
     study's dataset-wide hardcoded bounds. Monotone per column, so it does not change tuple
     identity.
  4. epochs<=40, early-stop patience 5, batch 64 vs the CDD study's 300 epochs / batch 8
     (h4_fusion_matched_budget.py runs the latter); the optimizer, CosineAnnealingWarmRestarts
     (T_0=25) scheduler, and inverse-frequency WeightedRandomSampler are the CDD study's,
     verbatim.

Run (each mode is explicit; bare invocation prints usage and trains nothing):
  TROPICAL_SCRATCH=/path/to/extraction python h4_fusion_collapse.py smoke   # forward-pass check
  TROPICAL_SCRATCH=/path/to/extraction python h4_fusion_collapse.py run     # the GPU run
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import albumentations as A
import timm
from albumentations.pytorch.transforms import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so the sibling import resolves
from leakage_collapse_gate import PANEL  # noqa: E402  the 9 panel cols — single source

SCRATCH = Path(os.environ["TROPICAL_SCRATCH"])  # no fallback; see README.md
HERE = Path(__file__).resolve().parent   # results land beside the script that
                                         # writes them: the path the paper reads
SUBSAMPLE = SCRATCH / "h4_subsample.parquet"     # the 40k membership + 2 split columns
PANEL_PARQUET = SCRATCH / "panel.parquet"        # the env table (one 9-col row per image)
OUT = HERE / "h4_fusion_collapse.csv"            # per (split, seed, arm) results (written by run)

# ---- the design; none tuned on any fold ----
N_CLASSES = 18                     # 17 pest and disease classes and a healthy class (h4_materialize_splits.py N_CLS_EXPECTED)
SEEDS = (0, 1, 2, 3, 4)            # 5 seeds
N_FOLDS = 5                        # outer test = h4_materialize_splits.py fold 0; inner val = fold 0 too
SUBSAMPLE_SEED = 0                 # structural — val carve + in-distribution shuffle; NOT a training seed
ARMS = ("img", "all")             # image-only model · late-logit fusion model
SPLITS = {"iid": None, "date": "pdate"}  # split -> group key (None == random)

# training procedure: the CDD study's optimizer/scheduler/sampler (BASELINE above); the reduced
# budget's epochs/batch/patience.
PX, BS, LR, BETAS = 256, 64, 1e-3, (0.9, 0.99)
EPOCHS, PATIENCE, T_0 = 40, 5, 25
DEV = f"cuda:{os.environ.get('TROPICAL_GPU', '0')}" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# model — src/model.py from github.com/hyunseoki/crop_disease @ c825d9d (the CDD study),
# the released file this script transcribes;
# num_classes 25 -> 18
# ----------------------------------------------------------------------------
class DaconLSTM(nn.Module):
    """The CDD study's src/model.py, verbatim. The 9-col panel as a length-1 sequence (B,1,9) -> 18
    logits; out[:,-1,:] is that single snapshot (deviation 1 in the header)."""

    def __init__(self, input_dim=9, hidden_dim=200, num_layers=1, output_dim=N_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 75)
        self.fc2 = nn.Linear(75, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc2(self.fc1(out[:, -1, :]))


class DaconModel(nn.Module):
    """The CDD study's src/model.py, verbatim. LATE LOGIT fusion of cnn->18 and lstm->18."""

    def __init__(self, cnn, rnn, num_classes=N_CLASSES):
        super().__init__()
        self.cnn, self.rnn = cnn, rnn
        self.dropout1, self.dropout2 = nn.Dropout(0.5), nn.Dropout(0.5)
        self.classifier = nn.Linear(num_classes * 2, num_classes)

    def forward(self, img, csv):
        out = torch.cat((self.dropout1(self.cnn(img)), self.dropout2(self.rnn(csv))), dim=1)
        return self.classifier(out)


def build_model(arm, pretrained=True):
    """src/main.py:105-117 analog. 'img' -> ResNet-50(18); 'all' -> DaconModel(cnn, DaconLSTM).
    `pretrained` is the training procedure's default; the smoke test passes False so it needs no network (the
    forward-pass plumbing is identical either way)."""
    cnn = timm.create_model("resnet50", pretrained=pretrained, num_classes=N_CLASSES)
    if arm == "img":
        return cnn.to(DEV)
    if arm == "all":
        return DaconModel(cnn, DaconLSTM()).to(DEV)
    raise ValueError(f"unknown arm {arm!r}")


# ----------------------------------------------------------------------------
# their augmentation (the CDD study's src/dataset.py:get_train_transforms), albumentations 2.x op
# names. Enters every model identically -> cancels in the estimand; matched so the absolute image number
# is comparable to a ResNet-50 baseline.
# ----------------------------------------------------------------------------
def train_tf():
    return A.Compose([
        A.Resize(PX, PX),
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(0.05, 0.05), fill=0, p=0.3),
        A.OneOf([  # the CDD study's A.Flip(p=0.5) (removed in albumentations 2.x): fire on 50%, then H/V/both
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Sequential([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=1.0),
        ], p=0.5),
        A.CoarseDropout(num_holes_range=(4, 16), hole_height_range=(25, 50),
                        hole_width_range=(25, 50), fill=0, p=0.5),
        A.Normalize(p=1.0), ToTensorV2(p=1.0),
    ])


def eval_tf():
    return A.Compose([A.Resize(PX, PX), A.Normalize(p=1.0), ToTensorV2(p=1.0)])


# ----------------------------------------------------------------------------
# data — subsample + panel join, class factorization, the split carve, normalization
# ----------------------------------------------------------------------------
def cls_map(sub):
    classes = sorted(sub["cls"].unique())
    assert len(classes) == N_CLASSES, f"expected {N_CLASSES} classes, got {len(classes)}"
    return {c: i for i, c in enumerate(classes)}  # deterministic 0..17 (sorted)


def load_metadata():
    """The 40k subsample joined to its 9-col panel; classes factorized to 0..17.
    Returns (sub, y, env_raw). env_raw is NaN-filled to 0 (deviation 2) but NOT normalized —
    normalization is fit per split on TRAIN rows (leakage-safe, deviation 3)."""
    sub = pd.read_parquet(SUBSAMPLE).reset_index(drop=True)
    assert len(sub) == 40000, f"subsample is {len(sub)} rows, expected 40000 (h4_materialize_splits.py)"
    panel = pd.read_parquet(PANEL_PARQUET)[["image"] + PANEL]
    m = sub.merge(panel, on="image", how="left", validate="one_to_one")
    assert len(m) == len(sub), "panel join changed the row count"
    no_panel = m[PANEL].isna().all(axis=1)
    assert not no_panel.any(), (  # a wholly-absent panel row breaks the join, not just a column
        f"{int(no_panel.sum())} subsample images have NO panel row — join incomplete")
    y = sub["cls"].map(cls_map(sub)).to_numpy(np.int64)
    env_raw = np.nan_to_num(m[PANEL].to_numpy(np.float32), nan=0.0)  # deviation 2 (the CDD study's convention)
    return sub, y, env_raw


def carve(sub, split):
    """Outer test = the split column (h4_materialize_splits.py fold 0). Inner val = fold 0 of a
    grouping-matched KFold/GroupKFold within TRAIN, so the checkpoint is selected on data
    disjoint from train by the SAME axis as the split (a date-disjoint val for the date split;
    h4_materialize_splits.py does not carve it). Deterministic in SUBSAMPLE_SEED (fixed across the 5
    training seeds). Returns positional index arrays (tr, va, te) into `sub`."""
    col = sub[f"split_{split}"].to_numpy()
    te = np.where(col == "test")[0]
    pool = np.where(col == "train")[0]
    gkey = SPLITS[split]
    if gkey is None:
        gen = KFold(N_FOLDS, shuffle=True, random_state=SUBSAMPLE_SEED).split(pool)
    else:
        groups = sub[gkey].astype(str).to_numpy()[pool]
        gen = GroupKFold(N_FOLDS).split(pool, groups=groups)
    tr_rel, va_rel = next(iter(gen))
    return pool[tr_rel], pool[va_rel], te


def normalize(env_raw, tr_idx):
    """Per-column min-max fit on TRAIN-proper rows, applied to all rows (deviation 3). A column
    that is constant on train maps to 0 (span->1), never NaN."""
    lo, hi = env_raw[tr_idx].min(0), env_raw[tr_idx].max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return ((env_raw - lo) / span).astype(np.float32)


class RandImgs:
    """Lazy random-image stand-in aligned to sub row order — the smoke test's pixel provider.
    Generates each HxWx3 uint8 image on demand (seeded per index, so it is deterministic and
    worker-safe) instead of allocating the full 40k x 256 x 256 x 3 (~7.5 GB) array. Module-level
    so the DataLoader workers can pickle it. NOT used by the run, which loads real imagery."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, j):
        return np.random.default_rng(int(j)).integers(0, 256, size=(PX, PX, 3), dtype=np.uint8)


class PanelDS(Dataset):
    """(image_tensor, env(1,9), y). `imgs` is a HxWx3 uint8 provider aligned to sub row order:
    the built cache for the run, a RandImgs stand-in for the smoke test. env is the normalized
    9-col panel as a length-1 sequence for the DaconLSTM."""

    def __init__(self, imgs, env, y, idx, tf, use_img, use_env):
        self.imgs, self.env, self.y, self.idx = imgs, env, y, idx
        self.tf, self.use_img, self.use_env = tf, use_img, use_env

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        img = self.tf(image=np.asarray(self.imgs[j]))["image"] if self.use_img else torch.zeros(1)
        env = torch.from_numpy(self.env[j]).unsqueeze(0) if self.use_env else torch.zeros(1)  # (1,9)
        return img, env, int(self.y[j])


def loader(imgs, env, y, idx, tf, arm, train, seed):
    """DataLoader; on train, the CDD study's inverse-frequency WeightedRandomSampler (src/util.py),
    seeded from `seed` alone so both models draw the identical order within a seed (pairs delta)."""
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


# ----------------------------------------------------------------------------
# train / eval — one training, best-val-macro-F1 checkpoint (the CDD study's model_best.pth)
# ----------------------------------------------------------------------------
def seed_everything(seed):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dl, arm):
    model.eval()
    ys, ps = [], []
    for img, env, yb in dl:
        with torch.autocast("cuda", enabled=DEV.startswith("cuda")):
            out = model(img.to(DEV), env.to(DEV)) if arm == "all" else model(img.to(DEV))
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    # macro-F1 over classes present in y_true (the CDD study's metric wherever a result is read;
    # keeps the img and all branches on one denominator so delta has no spurious term).
    labels = np.unique(y)
    return (f1_score(y, p, labels=labels, average="macro", zero_division=0),
            balanced_accuracy_score(y, p))


def train_arm(imgs, env_raw, y, tr, va, te, seed, arm):
    """One training: Adam + CosineAnnealingWarmRestarts, early-stop patience 5 on val macro-F1,
    best-val checkpoint. Normalization is fit on this model's TRAIN rows. Returns a metrics dict."""
    seed_everything(seed)
    env = normalize(env_raw, tr) if arm == "all" else env_raw  # the image-only model ignores env
    model = build_model(arm)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T_0, eta_min=LR / 1e3)
    lossf = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.startswith("cuda"))

    dl_tr = loader(imgs, env, y, tr, train_tf(), arm, True, seed)
    dl_va = loader(imgs, env, y, va, eval_tf(), arm, False, seed)

    t0 = time.time()
    best_val, best_state, best_ep, since = -1.0, None, -1, 0
    for ep in range(EPOCHS):
        model.train()
        for img, e, yb in dl_tr:
            yb = yb.to(DEV, non_blocking=True)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=DEV.startswith("cuda")):
                out = model(img.to(DEV), e.to(DEV)) if arm == "all" else model(img.to(DEV))
                loss = lossf(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        vf1, _ = evaluate(model, dl_va, arm)
        if vf1 > best_val:
            best_val, best_ep, best_state, since = vf1, ep, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= PATIENCE:
                break

    model.load_state_dict(best_state)
    tf1, tba = evaluate(model, loader(imgs, env, y, te, eval_tf(), arm, False, seed), arm)
    return dict(test_f1=tf1, test_ba=tba, val_f1=best_val, best_epoch=best_ep,
                n_train=len(tr), n_val=len(va), n_test=len(te),
                n_class_train=int(len(np.unique(y[tr]))), n_class_test=int(len(np.unique(y[te]))),
                wall_s=time.time() - t0)


def ci(x):
    """mean, lo, hi of a 95% t-interval."""
    x = np.asarray(x, float)
    m, sd = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))
    return m, m - h, m + h


# ----------------------------------------------------------------------------
# report — the criteria printed for the reduced-budget run, evaluated on the run CSV
# ----------------------------------------------------------------------------
def report(long):
    """long: rows of (split, seed, arm, test_f1, ...). Prints and returns whether the
    in-distribution gain replicates and whether the date-disjoint gain's CI includes zero."""
    w = long.pivot_table(index=["split", "seed"], columns="arm", values="test_f1").reset_index()
    w["delta"] = w["all"] - w["img"]
    summ = {}
    print("\n" + "=" * 88)
    print("MACRO F1 delta = all - img  (mean [95% CI] over the 5 seeds, paired within seed)")
    print("=" * 88)
    for s in ("iid", "date"):
        g = w[w.split == s]
        if not len(g):
            continue
        md, ld, hd = ci(g["delta"].values)
        summ[s] = dict(d=md, lo=ld, hi=hd)
        print(f"  {s:5s} delta={md:+.4f} [{ld:+.4f}, {hd:+.4f}]   "
              f"img={g['img'].mean():.4f} all={g['all'].mean():.4f}")

    print("\n" + "=" * 88)
    print("CRITERIA")
    print("=" * 88)
    replicates = "iid" in summ and summ["iid"]["lo"] > 0                      # in-distribution delta CI excludes 0
    date_includes_zero = "date" in summ and summ["date"]["lo"] < 0 < summ["date"]["hi"]  # date delta CI includes 0
    if "iid" in summ:
        print(f"  replicates: in-distribution delta CI excludes 0 : {replicates}  "
              f"(else no collapse can be claimed)")
    if "date" in summ:
        print(f"  date delta CI includes 0: {date_includes_zero}")
    if "iid" in summ and "date" in summ:
        head = summ["iid"]["d"] - summ["date"]["d"]
        print(f"  reduction = delta(iid) - delta(date) = {head:+.4f}")
    confirmed = replicates and date_includes_zero
    if "iid" in summ and "date" in summ:
        if confirmed:
            print("\n  a positive in-distribution fusion gain whose date-disjoint counterpart "
                  "is statistically\n  indistinguishable from zero.")
        elif not replicates:
            print("\n  the in-distribution fusion gain does not replicate on the AI Hub dataset; "
                  "claim NO collapse.")
        elif not date_includes_zero:
            print("\n  the gain remains above zero on the date-disjoint split: the panel "
                  "carries\n  signal beyond the timestamp, reported as such.")
    return dict(replicates=replicates, date_includes_zero=date_includes_zero, confirmed=confirmed, summ=summ)


# ----------------------------------------------------------------------------
# image cache — the preparation step that precedes the GPU run
# ----------------------------------------------------------------------------
def load_images(sub):
    """The (40k, PX, PX, 3) uint8 image cache aligned to sub row order, built by
    h4_build_image_cache.py, separate from this script and from the smoke test (which injects
    random images). Fail loud rather than fabricate pixels."""
    cache = SCRATCH / f"h4_img{PX}.npy"
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    raise NotImplementedError(
        f"image cache {cache} is missing. Build it with h4_build_image_cache.py before "
        f"the run. The smoke test does not need it.")


# ----------------------------------------------------------------------------
# smoke — verify the forward pass + the metadata/val-carve plumbing.
# Touches no training loop, no image file, no checkpoint.
# ----------------------------------------------------------------------------
def smoke():
    print("=" * 88)
    print(f"FUSION EXPERIMENT — FORWARD-PASS SMOKE TEST   (device {DEV}; trains nothing)")
    print("=" * 88)

    # (A) model forward + backward, both models, on correctly-shaped tensors (no images, no data).
    print("\n[A] model forward/backward on random tensors")
    B = 4
    for arm in ARMS:
        model = build_model(arm, pretrained=False)  # arch-identical; no network needed
        img = torch.randn(B, 3, PX, PX, device=DEV)
        env = torch.randn(B, 1, 9, device=DEV)      # the length-1 panel sequence
        yb = torch.randint(0, N_CLASSES, (B,), device=DEV)
        out = model(img, env) if arm == "all" else model(img)
        assert out.shape == (B, N_CLASSES), f"{arm}: got {tuple(out.shape)}, want {(B, N_CLASSES)}"
        loss = nn.CrossEntropyLoss()(out, yb)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), f"{arm}: bad/absent grads"
        print(f"    {arm:4s}  out {tuple(out.shape)}  loss {loss.item():.3f}  "
              f"grad tensors {len(grads)}  OK")

    # (B) the DaconLSTM must genuinely consume the 9-col panel (length-1 sequence -> 18 logits).
    print("\n[B] DaconLSTM on the length-1 panel sequence")
    lstm = DaconLSTM().to(DEV)
    o = lstm(torch.randn(7, 1, 9, device=DEV))
    assert o.shape == (7, N_CLASSES), o.shape
    print(f"    (7,1,9) -> {tuple(o.shape)}  OK")

    # (C) metadata plumbing on the REAL subsample: join, factorize, carve, normalize.
    print("\n[C] metadata plumbing on the 40k subsample (real data, no pixels)")
    sub, y, env_raw = load_metadata()
    assert y.min() == 0 and y.max() == N_CLASSES - 1 and len(np.unique(y)) == N_CLASSES
    print(f"    loaded {len(sub)} rows; y in [{y.min()},{y.max()}], {len(np.unique(y))} classes; "
          f"env_raw {env_raw.shape} finite={np.isfinite(env_raw).all()}")
    for split in SPLITS:
        tr, va, te = carve(sub, split)
        assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0
        # the val carve must select the checkpoint on all 18 classes and (date) on a group
        # disjoint from train — the point of carving it by the split's own axis.
        val_cls_ok = set(y[va]) <= set(y[tr])
        env = normalize(env_raw, tr)
        assert np.isfinite(env).all(), f"{split}: normalize produced non-finite env"
        gkey = SPLITS[split]
        if gkey is None:
            grp_note = "random"
        else:
            gv = sub[gkey].astype(str).to_numpy()
            tr_va_disjoint = len(set(gv[va]) & set(gv[tr])) == 0
            assert tr_va_disjoint, f"{split}: val shares a {gkey} group with train (leaked checkpoint)"
            grp_note = (f"{gkey}: val/train groups disjoint={tr_va_disjoint}, "
                        f"n_val_groups={len(set(gv[va]))}")
        print(f"    {split:5s} tr/va/te = {len(tr):5d}/{len(va):4d}/{len(te):4d}  "
              f"val⊆train classes={val_cls_ok}  cls_train={len(np.unique(y[tr]))} "
              f"cls_val={len(np.unique(y[va]))}  [{grp_note}]")

    # (D) one real batch of METADATA through the loader + the fusion model (random pixels only),
    #     and the train-time augmentation pipeline (the albumentations 2.x ops that could break).
    print("\n[D] one loader batch (random pixels, real env+labels) through the fusion model")
    assert train_tf()(image=RandImgs(1)[0])["image"].shape == (3, PX, PX)  # augment path runs
    tr, va, te = carve(sub, "date")
    env = normalize(env_raw, tr)
    dl = loader(RandImgs(len(sub)), env, y, te[:BS * 3], eval_tf(), "all", False, 0)
    img, e, yb = next(iter(dl))
    assert e.shape[1:] == (1, 9), f"env batch shape {tuple(e.shape)} != (B,1,9)"
    model = build_model("all", pretrained=False)
    out = model(img.to(DEV), e.to(DEV))
    assert out.shape[0] == img.shape[0] and out.shape[1] == N_CLASSES
    print(f"    train_tf augment OK; batch img {tuple(img.shape)} env {tuple(e.shape)} "
          f"-> logits {tuple(out.shape)}  OK")

    # (E) the criterion arithmetic on SYNTHETIC deltas (verifies report()'s logic before a run).
    #     Deltas are constructed with exact means (per-seed jitter sums to zero): in-distribution
    #     gain +0.06 (CI excludes 0), date gain 0 (CI straddles 0). The report
    #     block that prints below is SYNTHETIC — a self-test of report()'s logic, NOT a
    #     measured result (nothing has been trained).
    print("\n[E] criterion logic on synthetic deltas (no training) — self-test of report()")
    print("    (the MACRO-F1 block below is SYNTHETIC logic verification, not a result)")
    jit = np.array([-0.006, -0.003, 0.0, 0.003, 0.006])  # mean 0 -> exact synthetic delta means
    means = {"iid": 0.06, "date": 0.0}
    rows = []
    for s, md in means.items():
        for k, seed in enumerate(SEEDS):
            rows.append(dict(split=s, seed=seed, arm="img", test_f1=0.60))
            rows.append(dict(split=s, seed=seed, arm="all", test_f1=0.60 + md + jit[k]))
    verdict = report(pd.DataFrame(rows))
    assert verdict["replicates"] and verdict["date_includes_zero"] and verdict["confirmed"], verdict
    assert not report(pd.DataFrame(  # and the mirror: a surviving date gain must not read as included-zero
        [dict(split="date", seed=seed, arm=a, test_f1=0.60 + (0.06 if a == "all" else 0.0) + jit[k])
         for k, seed in enumerate(SEEDS) for a in ARMS]))["date_includes_zero"]
    print("    synthetic (in-distribution gain +0.06, date gain ~0) -> replicates & date CI includes 0; "
          "a surviving date gain -> date CI excludes 0  OK")

    print("\n" + "=" * 88)
    print("SMOKE PASS — model forward/backward, panel LSTM, split carve, loader, and the "
          "criterion\narithmetic all check out. Nothing trained.")
    print("=" * 88)


# ----------------------------------------------------------------------------
# run — the GPU run
# ----------------------------------------------------------------------------
def run():
    print("=" * 88)
    print("FUSION RUN — 2 models x 2 splits x 5 seeds.")
    print("=" * 88)
    sub, y, env_raw = load_metadata()
    imgs = load_images(sub)  # fail-loud if the image cache is not built (see load_images)
    rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
    done = {(r["split"], r["seed"], r["arm"]) for r in rows}
    if done:
        print(f"resuming: {len(done)} trainings already in {OUT}", flush=True)
    for split in SPLITS:
        tr, va, te = carve(sub, split)
        for seed in SEEDS:
            for arm in ARMS:
                if (split, seed, arm) in done:
                    continue
                r = train_arm(imgs, env_raw, y, tr, va, te, seed, arm)
                rows.append(dict(split=split, seed=seed, arm=arm, **r))
                print(f"  {split:5s} seed={seed} {arm:3s} f1={r['test_f1']:.4f} "
                      f"val={r['val_f1']:.4f} ep={r['best_epoch']} {r['wall_s'] / 60:.1f}min",
                      flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False)
    report(pd.DataFrame(rows))
    print(f"\nwrote {OUT}")


def main():
    ap = argparse.ArgumentParser(description="AI Hub fusion experiment at the reduced budget (40 epochs)")
    ap.add_argument("mode", nargs="?", choices=["smoke", "run"],
                    help="smoke: forward-pass + plumbing check (trains nothing). "
                         "run: the GPU run.")
    args = ap.parse_args()
    if args.mode == "smoke":
        smoke()
    elif args.mode == "run":
        run()
    else:
        ap.print_help()
        print("\nNo mode given — nothing run. `smoke` verifies the forward pass; `run` trains.")


if __name__ == "__main__":
    main()
