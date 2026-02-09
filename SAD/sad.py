# =========================
# sad.py (FULL, 1 FILE)
# SAD Dataset preprocessing + labeling (alert/drowsy)
# - works with folder containing only *.set files (like your screenshot)
# - preprocess: 0.5Hz HPF -> 50Hz LPF -> resample 250Hz -> ICA -> epoch
# - event mapping auto-search for codes like 251/252/253/254
# - labels by local RT + global RT with alert_rt (5th percentile)
# - per-session min samples constraint + augmentation
# =========================

import os
import glob
import math
import random
from collections import Counter, defaultdict

import numpy as np
import mne
from mne.preprocessing import ICA

# =========================
# CONFIG
# =========================
SAD_ROOT = r"E:\Duc\Safety Driving (2)\Safety Driving\SAD\DTA"  # <-- CHANGE THIS
SEED = 42

# Preprocess (flowchart)
HPF = 0.5
LPF = 50.0
TARGET_FS = 250

# ICA
ICA_METHOD = "fastica"
ICA_N_COMPONENTS = 0.99          # keep 99% variance
ICA_RANDOM_STATE = 42
ICA_MAX_AUTO_EXCLUDE = 15        # safety cap

# Epoch extraction
TMIN = -1.0
TMAX = 2.0
BASELINE = (None, 0.0)           # baseline until 0s

# RT labeling rules
GLOBAL_RT_WINDOW_SEC = 90.0
ALERT_MULT = 1.5
DROWSY_MULT = 2.5
ALERT_RT_PERCENTILE = 5          # 5th percentile of local RTs

# Valid RT sanity range (seconds) – helps mapping search
MIN_RT_SEC = 0.2
MAX_RT_SEC = 10.0

# Constraints (paper-like)
MIN_PER_CLASS_PER_SESSION = 50   # set 0 to disable session dropping

# Augmentation
USE_AUG = True
AUG_NOISE_STD = 0.01             # relative scale (w.r.t std of epoch)
AUG_TIME_SHIFT_MAX = 15          # samples shift at TARGET_FS
AUG_SCALE_RANGE = (0.9, 1.1)
AUG_REPEAT = 1                   # how many augmented copies per kept epoch (per class balancing handled below)

# Cache
CACHE_PATH = os.path.join(SAD_ROOT, f"SAD_cache_hp{HPF}_lp{LPF}_fs{TARGET_FS}_t{TMIN}_{TMAX}.npz")

# If you know event mapping exactly, you can override here:
# Example:
#   DEV_CODES = {"251", "252"}     # deviation onset (left/right)
#   RESP_CODES = {"253", "254"}    # response onset (left/right)
DEV_CODES = None
RESP_CODES = None

# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def make_sliding_epochs(raw_clean, dev_times_sec, label_per_trial,
                        win_sec=1.0, step_sec=0.5):
    """
    Paper-style:
    1 trial -> multiple EEG windows
    """
    sf = raw_clean.info["sfreq"]
    win = int(win_sec * sf)
    step = int(step_sec * sf)

    X_list, y_list = [], []

    data = raw_clean.get_data()  # (C, T)

    for t_dev, y in zip(dev_times_sec, label_per_trial):
        if y == -1:
            continue

        center = int(t_dev * sf)

        # window range: [-2s, 0s] before deviation (paper-style)
        start_min = center - int(2.0 * sf)
        start_max = center - win

        for s in range(start_min, start_max, step):
            if s < 0 or s + win > data.shape[1]:
                continue
            X_list.append(data[:, s:s + win])
            y_list.append(y)

    return np.asarray(X_list, np.float32), np.asarray(y_list, np.int64)

def list_set_files(root):
    files = sorted(glob.glob(os.path.join(root, "*.set")))
    return files


def get_subject_id_from_filename(fname):
    # s01_060227n.set -> s01
    base = os.path.basename(fname)
    return base.split("_")[0]


def load_eeglab_set(path):
    # MNE can read EEGLAB .set
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    return raw


def preprocess_raw(raw: mne.io.BaseRaw):
    # Ensure EEG picks
    raw = raw.copy()

    # Filter
    raw.filter(l_freq=HPF, h_freq=LPF, verbose=False)

    # Resample
    if abs(raw.info["sfreq"] - TARGET_FS) > 1e-6:
        raw.resample(TARGET_FS, npad="auto", verbose=False)

    # ICA
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, meg=False, stim=False, exclude="bads")
    ica = ICA(
        n_components=ICA_N_COMPONENTS,
        method=ICA_METHOD,
        random_state=ICA_RANDOM_STATE,
        max_iter="auto",
    )
    ica.fit(raw, picks=picks_eeg, verbose=False)

    # Auto-exclude artifacts (conservative heuristic):
    # - if EOG exists, use find_bads_eog
    # - else use kurtosis outliers on ICA sources
    exclude = []
    if "eog" in raw:
        try:
            eog_inds, _ = ica.find_bads_eog(raw, verbose=False)
            exclude.extend(eog_inds)
        except Exception:
            pass

    if len(exclude) == 0:
        # kurtosis heuristic
        sources = ica.get_sources(raw).get_data()  # (n_comp, T)
        # kurtosis (manual, stable)
        mean = sources.mean(axis=1, keepdims=True)
        std = sources.std(axis=1, keepdims=True) + 1e-8
        z = (sources - mean) / std
        kurt = np.mean(z**4, axis=1)
        # mark extreme outliers
        thr = np.quantile(kurt, 0.98)
        cand = np.where(kurt >= thr)[0].tolist()
        exclude = cand[:ICA_MAX_AUTO_EXCLUDE]

    ica.exclude = exclude
    raw_clean = ica.apply(raw.copy(), verbose=False)

    return raw_clean, ica


def get_annotations(raw):
    ann = raw.annotations
    desc = [str(d) for d in ann.description]
    onsets = ann.onset
    # durations = ann.duration (unused)
    return np.array(onsets, float), np.array(desc, object)


def print_unique_ann(raw, k=30):
    _, desc = get_annotations(raw)
    uniq = sorted(set(desc.tolist()))
    print("Unique annotation descriptions (first 30):", uniq[:k])


def pair_events_by_codes(onsets_sec, desc, dev_codes, resp_codes, max_rt=MAX_RT_SEC):
    """
    Pair each deviation onset with the nearest following response onset within max_rt seconds.
    Returns list of (dev_time, resp_time, local_rt).
    """
    dev_times = sorted([t for t, d in zip(onsets_sec, desc) if d in dev_codes])
    resp_times = sorted([t for t, d in zip(onsets_sec, desc) if d in resp_codes])

    pairs = []
    j = 0
    for dt in dev_times:
        while j < len(resp_times) and resp_times[j] <= dt:
            j += 1
        if j >= len(resp_times):
            break
        rt = resp_times[j] - dt
        if 0 < rt <= max_rt:
            pairs.append((dt, resp_times[j], rt))
    return pairs


def score_mapping(pairs):
    """
    Heuristic to choose correct mapping:
    prefer many valid pairs with RT in [MIN_RT_SEC, MAX_RT_SEC],
    with median RT in plausible range (e.g., 0.6~3.5).
    """
    if len(pairs) == 0:
        return -1e9
    rts = np.array([p[2] for p in pairs], float)
    valid = (rts >= MIN_RT_SEC) & (rts <= MAX_RT_SEC)
    frac = valid.mean()
    med = np.median(rts[valid]) if valid.any() else 999

    # score components
    score = 0.0
    score += len(pairs) * 1.0
    score += frac * 100.0
    # penalize very small median
    if med < 0.4:
        score -= 200.0
    if med > 6.0:
        score -= 100.0
    # prefer median around ~1-2s
    score -= abs(med - 1.5) * 10.0
    return score


def auto_find_event_mapping(raw):
    """
    For case with 4 unique codes like '251','252','253','254':
    try all splits (2 codes as deviation, 2 as response) and pick best.
    """
    onsets_sec, desc = get_annotations(raw)
    uniq = sorted(set(desc.tolist()))

    # keep only numeric-like codes (string)
    uniq = [u for u in uniq if u.isdigit()]
    if len(uniq) < 2:
        return None, None, None

    # If user already knows mapping:
    if DEV_CODES is not None and RESP_CODES is not None:
        dev = set([str(x) for x in DEV_CODES])
        resp = set([str(x) for x in RESP_CODES])
        pairs = pair_events_by_codes(onsets_sec, desc, dev, resp)
        return dev, resp, pairs

    # If 4 codes, brute force 2-2 split
    best = None
    best_score = -1e18
    best_pairs = None

    if len(uniq) == 4:
        codes = uniq
        # all combinations choose 2 as dev
        from itertools import combinations
        for dev_sel in combinations(codes, 2):
            dev = set(dev_sel)
            resp = set([c for c in codes if c not in dev])
            pairs = pair_events_by_codes(onsets_sec, desc, dev, resp)
            s = score_mapping(pairs)
            if s > best_score:
                best_score = s
                best = (dev, resp)
                best_pairs = pairs

        return best[0], best[1], best_pairs

    # Otherwise: fallback – pick most frequent 2 as dev, others as resp (weak)
    counts = Counter(desc.tolist())
    top = [c for c, _ in counts.most_common(4)]
    dev = set(top[:2])
    resp = set(top[2:4])
    pairs = pair_events_by_codes(onsets_sec, desc, dev, resp)
    return dev, resp, pairs


def compute_global_rt(dev_times, local_rts, window_sec=GLOBAL_RT_WINDOW_SEC):
    """
    global RT per trial = mean local RT within [dev_time - window_sec, dev_time)
    """
    dev_times = np.asarray(dev_times, float)
    local_rts = np.asarray(local_rts, float)

    g = np.zeros_like(local_rts)
    for i, t in enumerate(dev_times):
        mask = (dev_times >= (t - window_sec)) & (dev_times < t)
        if mask.any():
            g[i] = float(np.mean(local_rts[mask]))
        else:
            g[i] = float(np.mean(local_rts[: i + 1]))  # fallback
    return g


def label_trials(local_rt, global_rt):
    """
    Use alert_rt = 5th percentile of local RT
    alert if local&global < 1.5*alert_rt
    drowsy if local&global > 2.5*alert_rt
    else unlabeled (-1)
    """
    local_rt = np.asarray(local_rt, float)
    global_rt = np.asarray(global_rt, float)

    alert_rt = np.percentile(local_rt, ALERT_RT_PERCENTILE)
    y = np.full(len(local_rt), -1, dtype=np.int64)

    alert_mask = (local_rt < ALERT_MULT * alert_rt) & (global_rt < ALERT_MULT * alert_rt)
    drowsy_mask = (local_rt > DROWSY_MULT * alert_rt) & (global_rt > DROWSY_MULT * alert_rt)

    y[alert_mask] = 0
    y[drowsy_mask] = 1
    return y, float(alert_rt)


def make_epochs(raw_clean, dev_times_sec):
    """
    Extract epochs around dev onset times (in seconds).
    """
    sf = raw_clean.info["sfreq"]
    events = []
    for t in dev_times_sec:
        sample = int(round(t * sf))
        events.append([sample, 0, 1])  # event_id=1
    events = np.asarray(events, int)
    event_id = dict(dev=1)

    picks_eeg = mne.pick_types(raw_clean.info, eeg=True, eog=False, meg=False, stim=False, exclude="bads")

    epochs = mne.Epochs(
        raw_clean,
        events=events,
        event_id=event_id,
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        preload=True,
        picks=picks_eeg,
        verbose=False,
    )
    X = epochs.get_data().astype(np.float32)  # (N, C, T)
    return X


def augment_epochs(X, y):
    """
    Simple augmentation for EEG epochs:
    - noise
    - time shift
    - scaling
    """
    if not USE_AUG:
        return X, y

    rng = np.random.RandomState(SEED)
    X_aug = [X]
    y_aug = [y]

    for _ in range(AUG_REPEAT):
        Xa = X.copy()

        # scaling
        scale = rng.uniform(AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1], size=(Xa.shape[0], 1, 1)).astype(np.float32)
        Xa = Xa * scale

        # noise (relative to epoch std)
        std = Xa.std(axis=(1, 2), keepdims=True) + 1e-8
        noise = rng.randn(*Xa.shape).astype(np.float32) * (AUG_NOISE_STD * std)
        Xa = Xa + noise

        # time shift
        max_shift = int(AUG_TIME_SHIFT_MAX)
        if max_shift > 0:
            shifts = rng.randint(-max_shift, max_shift + 1, size=(Xa.shape[0],))
            for i, s in enumerate(shifts):
                if s == 0:
                    continue
                Xa[i] = np.roll(Xa[i], shift=s, axis=-1)

        X_aug.append(Xa)
        y_aug.append(y.copy())

    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


def sliding_windows_per_trial(raw, dev_times, labels,
                              win_sec=1.0,
                              step_sec=0.5,
                              remember_sec=2.0):
    """
    Paper-faithful:
    - take EEG before deviation onset
    - sliding windows
    - SAME label as the trial
    """
    sf = raw.info["sfreq"]
    win = int(win_sec * sf)
    step = int(step_sec * sf)
    remember = int(remember_sec * sf)

    data = raw.get_data()  # (C, T)

    X_list, y_list = [], []

    for t, y in zip(dev_times, labels):
        center = int(t * sf)
        start_min = center - remember
        start_max = center - win

        for s in range(start_min, start_max + 1, step):
            if s < 0 or s + win > data.shape[1]:
                continue
            X_list.append(data[:, s:s + win])
            y_list.append(y)

    return np.asarray(X_list, np.float32), np.asarray(y_list, np.int64)

def process_single_session(set_path):
    subject = get_subject_id_from_filename(set_path)
    print(f"\n=== Session: {os.path.basename(set_path)} | subject={subject} ===")

    raw = load_eeglab_set(set_path)
    print_unique_ann(raw)

    # ---------- event mapping ----------
    dev_codes, resp_codes, pairs = auto_find_event_mapping(raw)
    if dev_codes is None or resp_codes is None or len(pairs) < 5:
        print("[SKIP SESSION] event parse failed")
        return None

    rts = np.array([p[2] for p in pairs], float)
    print(f"[EVENT MAP] dev={sorted(dev_codes)} | resp={sorted(resp_codes)} "
          f"| pairs={len(pairs)} | RT median={np.median(rts):.3f}s")

    # ---------- preprocess ----------
    raw_clean, ica = preprocess_raw(raw)
    print(f"Preprocess OK: fs={raw_clean.info['sfreq']} | ICA excluded comps={len(ica.exclude)}")

    # ---------- RT labeling ----------
    dev_times = np.array([p[0] for p in pairs])
    local_rt = np.array([p[2] for p in pairs])
    global_rt = compute_global_rt(dev_times, local_rt)

    y_trial, alert_rt = label_trials(local_rt, global_rt)
    keep = (y_trial != -1)

    print(f"Trials total: {len(pairs)} | labeled={keep.sum()} | alert_rt={alert_rt:.4f}s")

    if keep.sum() == 0:
        return None

    # ---------- SLIDING WINDOWS (PAPER CORE) ----------
    X, y = sliding_windows_per_trial(
        raw_clean,
        dev_times[keep],
        y_trial[keep],
        win_sec=1.0,
        step_sec=0.5,
        remember_sec=8.0
    )

    dist = Counter(y.tolist())
    print(f"Epoch windows: {len(y)} | label dist={dict(dist)}")

    return subject, X, y


def maybe_balance_and_augment(X, y):
    """
    Balance classes by augmenting minority class (within session).
    """
    if not USE_AUG:
        return X, y

    dist = Counter(y.tolist())
    c0, c1 = dist.get(0, 0), dist.get(1, 0)
    if c0 == 0 or c1 == 0:
        return X, y

    # augment both (light), then oversample minority to match majority
    X_aug, y_aug = augment_epochs(X, y)

    # oversample minority to match majority (on augmented pool)
    dist2 = Counter(y_aug.tolist())
    maj = 0 if dist2[0] >= dist2[1] else 1
    minc = 1 - maj
    n_maj = dist2[maj]
    n_min = dist2[minc]

    if n_min < n_maj:
        idx_min = np.where(y_aug == minc)[0]
        need = n_maj - n_min
        rep = np.random.choice(idx_min, size=need, replace=True)
        X_bal = np.concatenate([X_aug, X_aug[rep]], axis=0)
        y_bal = np.concatenate([y_aug, y_aug[rep]], axis=0)
        return X_bal, y_bal

    return X_aug, y_aug


def build_sad_subject_trials(root):
    """
    Paper-faithful SAD pipeline:
    - Process ALL sessions
    - Label trials (alert / drowsy) per session
    - MERGE by SUBJECT (paper keeps 13 subjects)
    - DROP SUBJECT if <50 per class AFTER merge
    - NO class balancing 1:1 (paper does NOT do that)
    """

    subject_trials = defaultdict(list)
    subject_labels = defaultdict(list)

    set_files = sorted([f for f in os.listdir(root) if f.lower().endswith(".set")])
    if len(set_files) == 0:
        raise RuntimeError(f"No .set files found under {root}")

    for fname in set_files:
        fpath = os.path.join(root, fname)

        try:
            ret = process_single_session(fpath)  # ✅ MUST USE THIS (EEG thật)
            if ret is None:
                continue
            subj, X, y = ret
        except Exception as e:
            print("[SKIP SESSION]", e)
            continue

        if len(y) == 0:
            print("[SKIP] no labeled trials")
            continue

        subject_trials[subj].append(X)
        subject_labels[subj].append(y)

    # -------- merge per subject --------
    subjects = []
    subjects_X = []
    subjects_Y = []

    total_counter = Counter()

    for subj in sorted(subject_trials.keys()):
        X = np.concatenate(subject_trials[subj], axis=0)
        y = np.concatenate(subject_labels[subj], axis=0)

        counter = Counter(y.tolist())
        print(f"\n[SUBJECT {subj}] merged trials={len(y)} | dist={dict(counter)}")

        # ✅ PAPER RULE: apply after merge
        if counter.get(0, 0) < 50 or counter.get(1, 0) < 50:
            print("[DROP SUBJECT] <50 samples per class AFTER merge")
            continue

        subjects.append(subj)
        subjects_X.append(X)
        subjects_Y.append(y)
        total_counter.update(counter)

    if len(subjects) == 0:
        raise RuntimeError("No valid subjects produced after merge. Check event mapping / labeling.")

    print("\n=== FINAL DATASET SUMMARY ===")
    print("Subjects kept:", len(subjects))
    print("Total samples:", sum(total_counter.values()))
    print("Label dist:", dict(total_counter))

    return subjects, subjects_X, subjects_Y




def summarize_dataset(subjects, subjects_X, subjects_Y):
    total_counter = Counter()
    per_subject = {}
    total_samples = 0

    for s, X, y in zip(subjects, subjects_X, subjects_Y):
        c = Counter(y.tolist())
        per_subject[s] = dict(c)
        total_counter.update(c)
        total_samples += len(y)

    print("\n========== DATASET SUMMARY ==========")
    print("Subjects:", len(subjects), subjects)
    print("Total samples:", total_samples)
    print("Total label counts:", dict(total_counter))
    print("Per-subject label counts:")
    for s in subjects:
        print(f"  {s}: {per_subject[s]}")
    print("=====================================\n")

    return total_counter, per_subject, total_samples


def save_cache(path, subjects, subjects_X, subjects_Y, total_counter, per_subject_counter):
    np.savez_compressed(
        path,
        subjects=np.array(subjects, dtype=object),
        subjects_X=np.array(subjects_X, dtype=object),
        subjects_Y=np.array(subjects_Y, dtype=object),
        total_counter=np.array(dict(total_counter), dtype=object),
        per_subject_counter=np.array(per_subject_counter, dtype=object),
    )


def load_cache(path):
    data = np.load(path, allow_pickle=True)
    subjects = list(data["subjects"])
    subjects_X = list(data["subjects_X"])
    subjects_Y = list(data["subjects_Y"])

    # backward compatible
    if "total_counter" in data.files:
        total_counter = Counter(data["total_counter"].item())
    else:
        total_counter = Counter()
        for y in subjects_Y:
            total_counter.update(Counter(np.asarray(y).tolist()))

    if "per_subject_counter" in data.files:
        per_subject_counter = data["per_subject_counter"].item()
    else:
        per_subject_counter = {}

    return subjects, subjects_X, subjects_Y, total_counter, per_subject_counter



# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)

    if os.path.exists(CACHE_PATH):
        print(f"[CACHE] Loading: {CACHE_PATH}")
        subjects, subjects_X, subjects_Y, total_counter, per_subject_counter = load_cache(CACHE_PATH)
        total_samples = sum(len(y) for y in subjects_Y)
        print("\n[CACHE] Loaded OK.")
        print("Subjects:", len(subjects))
        print("Total samples:", total_samples)
        print("Total label counts:", dict(total_counter))
        return

    subjects, subjects_X, subjects_Y = build_sad_subject_trials(SAD_ROOT)

    total_counter, per_subject_counter, _ = summarize_dataset(subjects, subjects_X, subjects_Y)

    print(f"[CACHE] Saving: {CACHE_PATH}")
    save_cache(CACHE_PATH, subjects, subjects_X, subjects_Y, total_counter, per_subject_counter)
    print("[DONE] Saved cache successfully.")


if __name__ == "__main__":
    main()
