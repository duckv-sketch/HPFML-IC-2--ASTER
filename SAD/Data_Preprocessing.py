# =========================
# sad_paper_label_build.py  (FULL, 1 FILE)
#
# Paper-faithful SAD labeling (as in your excerpt):
# - event types: 251/252(dev), 253(resp onset), 254(resp offset)
# - local RT (RTl) = resp_onset - dev_onset
# - global RT (RTg) = mean(local RT) of trials within 90s BEFORE current dev_onset
# - alertRT (RTa) = 5th percentile of local RT (per SESSION)
# - labeling (Eq.16 in your excerpt):
#     y=0 (fatigue)     if RTl > 2.5*RTa AND RTg > 2.5*RTa
#     y=1 (nonfatigue)  if RTl < 1.5*RTa AND RTg < 1.5*RTa
#
# Preprocess (matching excerpt as close as possible in Python/MNE):
# - band-pass 1–50 Hz
# - downsample to 128 Hz
# - (EEGLab AAR + visual checking can't be perfectly replicated here; we use ICA-based removal)
#
# Trials:
# - "3s EEG before onset" => epoch [-3, 0] seconds relative to deviation onset
#
# Filtering (matching excerpt):
# - keep subjects where the SMALLER class trial count >= 50 (subject-level)
# - do NOT balance classes (keep unbalanced)
#
# Optional augmentation:
# - augmentation is for training convenience; by default it is applied AFTER subject filtering
# - you can choose whether augmentation contributes to min-count filtering via config.
# =========================

import os
import glob
import re
import random
from collections import Counter, defaultdict
from typing import Optional, Tuple, List, Dict

import numpy as np
import mne
from mne.preprocessing import ICA

# =========================
# CONFIG
# =========================
SAD_ROOT = r"E:\Duc\Safety Driving (2)\Safety Driving\SAD"  # <-- CHANGE THIS
SEED = 42

# Preprocess (paper-like)
HPF = 1.0
LPF = 50.0
TARGET_FS = 128

# ICA (approximate ocular/muscular removal)
ICA_METHOD = "fastica"
ICA_N_COMPONENTS = 0.99
ICA_RANDOM_STATE = 42
ICA_MAX_AUTO_EXCLUDE = 15

# Epoch extraction: 3s before lane departure event onset
TMIN = -3.0
TMAX = 0.0
BASELINE = None  # paper doesn't specify baseline correction

# RT labeling rules
GLOBAL_RT_WINDOW_SEC = 90.0
ALERT_MULT = 1.5
DROWSY_MULT = 2.5
ALERT_RT_PERCENTILE = 5

# sanity RT range
MIN_RT_SEC = 0.2
MAX_RT_SEC = 10.0

# Filtering (paper statement)
MIN_PER_CLASS_PER_SUBJECT = 50  # "smaller class trial > 50" -> use >= 50

# Optional: also filter very bad sessions (not in excerpt; default off)
FILTER_SESSION_MIN_PAIRS = False
MIN_VALID_PAIRS_PER_SESSION = 30

# Augmentation (optional)
USE_AUG = True
AUG_REPEAT = 1                 # total multiplier = (1 + AUG_REPEAT). 1 -> x2
AUG_NOISE_STD = 0.01
AUG_TIME_SHIFT_MAX = 10
AUG_SCALE_RANGE = (0.9, 1.1)

# IMPORTANT:
# If True, the subject min-count filtering uses augmented counts (NOT paper-faithful).
# If False, filtering uses RAW labeled trials (paper-faithful). Recommended: False.
APPLY_SUBJECT_MINCOUNT_AFTER_AUG = False

# Event codes
DEV_CODES = {"251", "252"}
RESP_ONSET_CODES = {"253"}
RESP_OFFSET_CODES = {"254"}    # fallback only

CACHE_PATH = os.path.join(
    SAD_ROOT,
    f"SAD_cache_paper_hp{HPF}_lp{LPF}_fs{TARGET_FS}_t{TMIN}_{TMAX}"
    f"_aug{int(USE_AUG)}x{AUG_REPEAT}_minAfterAug{int(APPLY_SUBJECT_MINCOUNT_AFTER_AUG)}.npz"
)

# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def list_set_files(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "*.set")))

def get_subject_id_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    # common naming: "subjXX_*.set"
    return base.split("_")[0]

def load_eeglab_set(path: str) -> mne.io.BaseRaw:
    return mne.io.read_raw_eeglab(path, preload=True, verbose=False)

def normalize_code(desc: str) -> Optional[str]:
    """
    Robustly extract digits from annotation description:
    handles 'S 251', '251.0', ' 251 ', etc.
    """
    if desc is None:
        return None
    s = str(desc).strip()
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else None

def get_events_from_annotations(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, np.ndarray]:
    ann = raw.annotations
    onsets = np.array(ann.onset, float)
    codes = []
    for d in ann.description:
        c = normalize_code(d)
        codes.append(c if c is not None else str(d).strip())
    return onsets, np.array(codes, dtype=object)

def preprocess_raw(raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, ICA]:
    raw = raw.copy()

    # bandpass
    raw.filter(l_freq=HPF, h_freq=LPF, verbose=False)

    # resample
    if abs(raw.info["sfreq"] - TARGET_FS) > 1e-6:
        raw.resample(TARGET_FS, npad="auto", verbose=False)

    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, meg=False, stim=False, exclude="bads")

    ica = ICA(
        n_components=ICA_N_COMPONENTS,
        method=ICA_METHOD,
        random_state=ICA_RANDOM_STATE,
        max_iter="auto",
    )
    ica.fit(raw, picks=picks_eeg, verbose=False)

    exclude = []
    # try eog-like detection if EOG channels exist
    try:
        eog_inds, _ = ica.find_bads_eog(raw, verbose=False)
        exclude.extend(eog_inds)
    except Exception:
        pass

    # fallback: high-kurtosis components
    if len(exclude) == 0:
        sources = ica.get_sources(raw).get_data()
        mean = sources.mean(axis=1, keepdims=True)
        std = sources.std(axis=1, keepdims=True) + 1e-8
        z = (sources - mean) / std
        kurt = np.mean(z**4, axis=1)
        thr = np.quantile(kurt, 0.98)
        cand = np.where(kurt >= thr)[0].tolist()
        exclude = cand[:ICA_MAX_AUTO_EXCLUDE]

    ica.exclude = exclude
    raw_clean = ica.apply(raw.copy(), verbose=False)
    return raw_clean, ica

def pair_deviation_to_response(
    onsets_sec: np.ndarray,
    codes: np.ndarray,
    dev_codes: set,
    resp_codes: set,
) -> List[Tuple[float, float, float]]:
    """
    Pair each deviation onset with the next response onset strictly after it.
    """
    dev_times = sorted([t for t, c in zip(onsets_sec, codes) if c in dev_codes])
    resp_times = sorted([t for t, c in zip(onsets_sec, codes) if c in resp_codes])

    pairs = []
    j = 0
    for dt in dev_times:
        while j < len(resp_times) and resp_times[j] <= dt:
            j += 1
        if j >= len(resp_times):
            break
        rt = resp_times[j] - dt
        if MIN_RT_SEC <= rt <= MAX_RT_SEC:
            pairs.append((dt, resp_times[j], rt))
            j += 1
    return pairs

def compute_global_rt(dev_times: np.ndarray, local_rts: np.ndarray, window_sec: float = GLOBAL_RT_WINDOW_SEC) -> np.ndarray:
    """
    RTg for trial i = mean RTl of all trials with dev_times in [t-window, t).
    If none exist, use mean of previous RTl (or first RTl if i==0).
    """
    dev_times = np.asarray(dev_times, float)
    local_rts = np.asarray(local_rts, float)
    g = np.zeros_like(local_rts, dtype=float)

    for i, t in enumerate(dev_times):
        mask = (dev_times >= (t - window_sec)) & (dev_times < t)
        if mask.any():
            g[i] = float(np.mean(local_rts[mask]))
        else:
            g[i] = float(np.mean(local_rts[:i])) if i > 0 else float(local_rts[0])
    return g

def label_trials(local_rt: np.ndarray, global_rt: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Paper Eq.(16):
      y=0 fatigue     if RTl > 2.5*RTa and RTg > 2.5*RTa
      y=1 nonfatigue  if RTl < 1.5*RTa and RTg < 1.5*RTa
    Unlabeled -> -1
    """
    local_rt = np.asarray(local_rt, float)
    global_rt = np.asarray(global_rt, float)
    alert_rt = float(np.percentile(local_rt, ALERT_RT_PERCENTILE))

    y = np.full(len(local_rt), -1, dtype=np.int64)

    nonfatigue_mask = (local_rt < ALERT_MULT * alert_rt) & (global_rt < ALERT_MULT * alert_rt)
    fatigue_mask    = (local_rt > DROWSY_MULT * alert_rt) & (global_rt > DROWSY_MULT * alert_rt)

    y[fatigue_mask] = 0
    y[nonfatigue_mask] = 1
    return y, alert_rt

def extract_epochs(raw_clean: mne.io.BaseRaw, dev_times_sec: np.ndarray) -> np.ndarray:
    sf = raw_clean.info["sfreq"]
    events = np.array([[int(round(t * sf)), 0, 1] for t in dev_times_sec], dtype=int)
    picks_eeg = mne.pick_types(raw_clean.info, eeg=True, eog=False, meg=False, stim=False, exclude="bads")

    epochs = mne.Epochs(
        raw_clean,
        events=events,
        event_id={"dev": 1},
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        preload=True,
        picks=picks_eeg,
        verbose=False,
    )
    return epochs.get_data().astype(np.float32)

def augment_trials(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    if (not USE_AUG) or (AUG_REPEAT <= 0):
        return X, y

    rng = np.random.RandomState(seed)
    X_list = [X]
    y_list = [y]

    for _ in range(AUG_REPEAT):
        Xa = X.copy()

        # scaling
        scale = rng.uniform(AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1], size=(Xa.shape[0], 1, 1)).astype(np.float32)
        Xa *= scale

        # noise (relative to per-sample std)
        std = Xa.std(axis=(1, 2), keepdims=True) + 1e-8
        Xa += rng.randn(*Xa.shape).astype(np.float32) * (AUG_NOISE_STD * std)

        # time shift (roll)
        max_shift = int(AUG_TIME_SHIFT_MAX)
        if max_shift > 0:
            shifts = rng.randint(-max_shift, max_shift + 1, size=(Xa.shape[0],))
            for i, s in enumerate(shifts):
                if s != 0:
                    Xa[i] = np.roll(Xa[i], shift=s, axis=-1)

        X_list.append(Xa)
        y_list.append(y.copy())

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

# =========================
# Processing
# =========================
def process_single_session(set_path: str):
    subject = get_subject_id_from_filename(set_path)
    print(f"\n=== Session: {os.path.basename(set_path)} | subject={subject} ===")

    raw = load_eeglab_set(set_path)
    onsets_sec, codes = get_events_from_annotations(raw)

    cnt = Counter(codes.tolist())
    print("[EVENT COUNT]", {k: cnt.get(k, 0) for k in ["251", "252", "253", "254"]})

    # Use 253 for response onset; fallback to 254 only if 253 absent
    resp_codes = set(RESP_ONSET_CODES)
    if cnt.get("253", 0) == 0 and cnt.get("254", 0) > 0:
        resp_codes = set(RESP_OFFSET_CODES)
        print("[WARN] No 253 found; fallback use 254 as response marker (check dataset!)")

    pairs = pair_deviation_to_response(onsets_sec, codes, DEV_CODES, resp_codes)

    if FILTER_SESSION_MIN_PAIRS and len(pairs) < MIN_VALID_PAIRS_PER_SESSION:
        print("[SKIP] too few valid RT pairs:", len(pairs))
        return None

    if len(pairs) == 0:
        print("[SKIP] no valid RT pairs")
        return None

    dev_times = np.array([p[0] for p in pairs], float)
    local_rt = np.array([p[2] for p in pairs], float)
    global_rt = compute_global_rt(dev_times, local_rt, window_sec=GLOBAL_RT_WINDOW_SEC)

    y_trial, alert_rt = label_trials(local_rt, global_rt)
    keep = (y_trial != -1)

    if keep.sum() == 0:
        print("[SKIP] no labeled trials (all -1)")
        return None

    # preprocess + epoch extraction on kept trials
    raw_clean, _ = preprocess_raw(raw)
    X = extract_epochs(raw_clean, dev_times[keep])
    y = y_trial[keep].astype(np.int64)

    dist = Counter(y.tolist())
    print(f"[LABEL] RTa={alert_rt:.4f}s | labeled={len(y)} | dist={dict(dist)}")

    return subject, X, y

def build_dataset(root: str):
    set_files = list_set_files(root)
    if len(set_files) == 0:
        raise RuntimeError(f"No .set files found under {root}")

    subject_X_raw: Dict[str, List[np.ndarray]] = defaultdict(list)
    subject_y_raw: Dict[str, List[np.ndarray]] = defaultdict(list)

    kept_sessions = 0
    for f in set_files:
        try:
            ret = process_single_session(f)
        except Exception as e:
            print("[SKIP SESSION] exception:", e)
            continue
        if ret is None:
            continue

        subj, X, y = ret
        subject_X_raw[subj].append(X)
        subject_y_raw[subj].append(y)
        kept_sessions += 1

    print(f"\n=== Sessions processed/kept: {kept_sessions} / {len(set_files)} ===")

    subjects = []
    subjects_X = []
    subjects_Y = []
    total_counter = Counter()

    for subj in sorted(subject_X_raw.keys()):
        X_raw = np.concatenate(subject_X_raw[subj], axis=0)
        y_raw = np.concatenate(subject_y_raw[subj], axis=0)
        dist_raw = Counter(y_raw.tolist())

        # subject-level filtering per excerpt
        if (dist_raw.get(0, 0) < MIN_PER_CLASS_PER_SUBJECT) or (dist_raw.get(1, 0) < MIN_PER_CLASS_PER_SUBJECT):
            print(f"[DROP SUBJECT] {subj} <{MIN_PER_CLASS_PER_SUBJECT}/class (RAW) | dist={dict(dist_raw)}")
            continue

        # optional augmentation (default AFTER filtering)
        if USE_AUG:
            X_aug, y_aug = augment_trials(X_raw, y_raw, seed=SEED)

            if APPLY_SUBJECT_MINCOUNT_AFTER_AUG:
                dist_aug = Counter(y_aug.tolist())
                if (dist_aug.get(0, 0) < MIN_PER_CLASS_PER_SUBJECT) or (dist_aug.get(1, 0) < MIN_PER_CLASS_PER_SUBJECT):
                    print(f"[DROP SUBJECT] {subj} <{MIN_PER_CLASS_PER_SUBJECT}/class (AUG) | dist={dict(dist_aug)}")
                    continue
                X_final, y_final, dist_final = X_aug, y_aug, dist_aug
            else:
                X_final, y_final, dist_final = X_aug, y_aug, Counter(y_aug.tolist())
        else:
            X_final, y_final, dist_final = X_raw, y_raw, dist_raw

        subjects.append(subj)
        subjects_X.append(X_final)
        subjects_Y.append(y_final)
        total_counter.update(dist_final)

        print(f"[SUBJECT {subj}] kept | samples={len(y_final)} | dist={dict(dist_final)}")

    if len(subjects) == 0:
        raise RuntimeError("No valid subjects after filtering. Check events/labels/config.")

    print("\n=== FINAL SUMMARY ===")
    print("Subjects kept:", len(subjects), subjects)
    print("Total samples:", sum(len(y) for y in subjects_Y))
    print("Total dist:", dict(total_counter))
    return subjects, subjects_X, subjects_Y, total_counter

# =========================
# Cache IO
# =========================
def save_cache(path: str, subjects, subjects_X, subjects_Y, total_counter):
    np.savez_compressed(
        path,
        subjects=np.array(subjects, dtype=object),
        subjects_X=np.array(subjects_X, dtype=object),
        subjects_Y=np.array(subjects_Y, dtype=object),
        total_counter=np.array(dict(total_counter), dtype=object),
        config=np.array({
            "HPF": HPF, "LPF": LPF, "TARGET_FS": TARGET_FS,
            "TMIN": TMIN, "TMAX": TMAX, "BASELINE": BASELINE,
            "GLOBAL_RT_WINDOW_SEC": GLOBAL_RT_WINDOW_SEC,
            "ALERT_MULT": ALERT_MULT, "DROWSY_MULT": DROWSY_MULT, "ALERT_RT_PERCENTILE": ALERT_RT_PERCENTILE,
            "MIN_RT_SEC": MIN_RT_SEC, "MAX_RT_SEC": MAX_RT_SEC,
            "MIN_PER_CLASS_PER_SUBJECT": MIN_PER_CLASS_PER_SUBJECT,
            "USE_AUG": USE_AUG, "AUG_REPEAT": AUG_REPEAT,
            "APPLY_SUBJECT_MINCOUNT_AFTER_AUG": APPLY_SUBJECT_MINCOUNT_AFTER_AUG,
        }, dtype=object),
    )

def load_cache(path: str):
    data = np.load(path, allow_pickle=True)
    subjects = list(data["subjects"])
    subjects_X = list(data["subjects_X"])
    subjects_Y = list(data["subjects_Y"])
    total_counter = Counter(data["total_counter"].item()) if "total_counter" in data.files else Counter()
    cfg = data["config"].item() if "config" in data.files else {}
    return subjects, subjects_X, subjects_Y, total_counter, cfg

# =========================
# Main
# =========================
def main():
    set_seed(SEED)

    if os.path.exists(CACHE_PATH):
        print(f"[CACHE] Loading: {CACHE_PATH}")
        subjects, subjects_X, subjects_Y, total_counter, cfg = load_cache(CACHE_PATH)
        print("[CACHE] OK")
        print("Config:", cfg)
        print("Subjects:", len(subjects), subjects)
        print("Total samples:", sum(len(y) for y in subjects_Y))
        print("Total dist:", dict(total_counter))
        return

    subjects, subjects_X, subjects_Y, total_counter = build_dataset(SAD_ROOT)

    print(f"\n[CACHE] Saving: {CACHE_PATH}")
    save_cache(CACHE_PATH, subjects, subjects_X, subjects_Y, total_counter)
    print("[DONE] cache saved")

if __name__ == "__main__":
    main()
