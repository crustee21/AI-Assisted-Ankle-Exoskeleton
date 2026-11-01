import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from pathlib import Path
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

RAW_RATE                = 1000
TARGET_RATE             = 100
DECIM                   = RAW_RATE // TARGET_RATE

WINDOW_MS               = 320
STRIDE_MS               = 10
WINDOW_SAMPLES          = round(WINDOW_MS * TARGET_RATE / 1000)
STRIDE_SAMPLES          = max(1, round(STRIDE_MS * TARGET_RATE / 1000))

LATE_STANCE_FRACTION    = 0.5
PRE_TOEOFF_MARGIN_MS    = 50

POSITIVE_OVERLAP_THRESHOLD = 0.6
NEGATIVE_OVERLAP_THRESHOLD = 0.1

DISCARD_AMBIGUOUS       = True

MIN_STANCE_MS           = 300
MAX_STANCE_MS           = 2000

CAUSAL                  = True
FILTER_ORDER            = 4

RECTIFY                 = 'full'
ENVELOPE_LP_HZ          = 30.0

GYRO_LP_HZ              = 30.0

MODE_WALKING            = 1
MODE_STANDING           = 6

MAX_NEG_TO_POS_RATIO    = 3.0

RANDOM_SEED             = 42

REPO_ROOT      = Path(__file__).resolve().parent
PROCESSED_ROOT = REPO_ROOT / "Processed_data"

SUBJECT_CIRCUITS = {
    "AB156": ("Processed1",  [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]),
    "AB185": ("Processed2",  [1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]),
    "AB186": ("Processed3",  [1, 3, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44, 46, 47, 48, 49]),
    "AB188": ("Processed4",  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 28, 29, 31, 35, 36, 37, 38, 39, 40]),
    "AB189": ("Processed5",  [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]),
    "AB190": ("Processed6",  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 19, 20, 21, 24, 25, 27, 28, 29, 30, 33, 34, 38, 39, 41, 42, 43, 44, 46, 47, 48]),
    "AB191": ("Processed7",  [3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 48]),
    "AB192": ("Processed8",  [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]),
    "AB193": ("Processed9",  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50]),
    "AB194": ("Processed10", [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]),
}

SUBJECT_FILES = {
    subject: [
        PROCESSED_ROOT / subfolder / f"{subject}_Circuit_{c:03d}_post.csv"
        for c in circuits
    ]
    for subject, (subfolder, circuits) in SUBJECT_CIRCUITS.items()
}

OUTPUT_DIR = REPO_ROOT / "enabl3s_dataset_sliding"

def apply_lowpass(signal, cutoff_hz, fs, order, causal):
    b, a = butter(order, cutoff_hz / (fs / 2.0), btype='low')
    return lfilter(b, a, signal) if causal else filtfilt(b, a, signal)

def rectify(signal, mode):
    return np.maximum(signal, 0.0) if mode == 'half' else np.abs(signal)

def emg_envelope(signal):
    return apply_lowpass(rectify(signal, RECTIFY), ENVELOPE_LP_HZ, RAW_RATE, FILTER_ORDER, CAUSAL)

def gyro_conditioned(signal):
    return apply_lowpass(signal, GYRO_LP_HZ, RAW_RATE, FILTER_ORDER, CAUSAL)

def zscore(signal, mu, std):
    return (signal - mu) if std < 1e-10 else (signal - mu) / std

def standing_stats(channel, mode, min_samples):
    standing = np.where(mode == MODE_STANDING)[0]
    if len(standing) < min_samples:
        return None

    breaks = np.where(np.diff(standing) > 1)[0]
    segments = []
    start = standing[0]
    for brk in breaks:
        segments.append((start, standing[brk]))
        start = standing[brk + 1]
    segments.append((start, standing[-1]))

    s, e = max(segments, key=lambda p: p[1] - p[0])
    seg = channel[s:e + 1]
    return float(np.mean(seg)), float(np.std(seg))

def build_intent_mask(df, toe_off_col, heel_contact_col):
    n = len(df)
    mode_vals = df['Mode'].values
    mask = np.zeros(n, dtype=bool)

    toe_offs      = df[toe_off_col][df[toe_off_col] > 0].values.astype(int)
    heel_contacts = df[heel_contact_col][df[heel_contact_col] > 0].values.astype(int)

    pre_margin = int(PRE_TOEOFF_MARGIN_MS * RAW_RATE / 1000)

    for to_idx in toe_offs:
        preceding = heel_contacts[heel_contacts < to_idx]
        if len(preceding) == 0:
            continue
        hc_idx = preceding[-1]
        stance = to_idx - hc_idx

        if stance < MIN_STANCE_MS or stance > MAX_STANCE_MS:
            continue
        if mode_vals[hc_idx] != MODE_WALKING or mode_vals[to_idx] != MODE_WALKING:
            continue

        region_start = to_idx - int(stance * LATE_STANCE_FRACTION)
        region_end   = to_idx - pre_margin
        region_start = max(region_start, 0)
        region_end   = min(region_end, n)
        if region_end > region_start:
            mask[region_start:region_end] = True

    return mask

def extract_from_circuit(df, subject_id):
    mode_raw = df['Mode'].values

    rta_env = emg_envelope(df['Right_TA'].values)
    lta_env = emg_envelope(df['Left_TA'].values)
    rgy     = gyro_conditioned(df['Right_Shank_Gy'].values)
    lgy     = gyro_conditioned(df['Left_Shank_Gy'].values)

    right_intent = build_intent_mask(df, 'Right_Toe_Off', 'Right_Heel_Contact')
    left_intent  = build_intent_mask(df, 'Left_Toe_Off',  'Left_Heel_Contact')

    sl = slice(None, None, DECIM)
    mode = mode_raw[sl]
    rta_env, lta_env = rta_env[sl], lta_env[sl]
    rgy, lgy         = rgy[sl], lgy[sl]
    right_intent, left_intent = right_intent[sl], left_intent[sl]
    n = len(mode)

    min_stand = max(1, int(0.2 * TARGET_RATE))
    norm = {}
    for name, ch in (('rta', rta_env), ('lta', lta_env), ('rgy', rgy), ('lgy', lgy)):
        st = standing_stats(ch, mode, min_stand)
        if st is None:
            return [], []
        norm[name] = zscore(ch, *st).astype(np.float32)

    windows, labels = [], []
    for ta_env, gy, intent_mask in (
        (norm['rta'], norm['rgy'], right_intent),
        (norm['lta'], norm['lgy'], left_intent),
    ):
        start = 0
        while start + WINDOW_SAMPLES <= n:
            end = start + WINDOW_SAMPLES
            win_modes = mode[start:end]

            first_mode = win_modes[0]
            if first_mode not in (MODE_WALKING, MODE_STANDING) or not np.all(win_modes == first_mode):
                start += STRIDE_SAMPLES
                continue

            overlap = np.sum(intent_mask[start:end]) / WINDOW_SAMPLES
            if overlap >= POSITIVE_OVERLAP_THRESHOLD:
                label = 1
            elif overlap <= NEGATIVE_OVERLAP_THRESHOLD:
                label = 0
            elif DISCARD_AMBIGUOUS:
                start += STRIDE_SAMPLES
                continue
            else:
                label = 1 if overlap >= 0.5 else 0

            windows.append(np.stack([ta_env[start:end], gy[start:end]], axis=0))
            labels.append(label)
            start += STRIDE_SAMPLES

    return windows, labels

def balance_negatives(X, y, groups, max_ratio, seed):
    if max_ratio is None:
        return X, y, groups

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    n_neg_keep = int(n_pos * max_ratio)

    if len(neg_idx) > n_neg_keep:
        neg_idx = rng.choice(neg_idx, size=n_neg_keep, replace=False)

    keep = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(keep)
    return X[keep], y[keep], groups[keep]

def main():
    global ENVELOPE_LP_HZ, GYRO_LP_HZ, OUTPUT_DIR
    ap = argparse.ArgumentParser(description="Extract causal, decimated sliding-window dataset.")
    ap.add_argument('--env-cutoff',  type=float, default=ENVELOPE_LP_HZ,
                    help='EMG envelope low-pass cutoff in Hz (sweep 20-40). Default %(default)s')
    ap.add_argument('--gyro-cutoff', type=float, default=GYRO_LP_HZ,
                    help='Gyro low-pass cutoff in Hz. Default %(default)s')
    ap.add_argument('--out',         type=str,   default=str(OUTPUT_DIR),
                    help='Output directory. Use a distinct dir per sweep point.')
    args = ap.parse_args()
    ENVELOPE_LP_HZ = args.env_cutoff
    GYRO_LP_HZ     = args.gyro_cutoff
    OUTPUT_DIR     = Path(args.out)

    nyq = TARGET_RATE / 2.0
    if ENVELOPE_LP_HZ >= nyq or GYRO_LP_HZ >= nyq:
        raise SystemExit(f"Cutoffs must be < TARGET_RATE/2 ({nyq} Hz); "
                         f"got env={ENVELOPE_LP_HZ}, gyro={GYRO_LP_HZ}.")
    print(f"Config: {RAW_RATE}Hz -> decim {DECIM} -> {TARGET_RATE}Hz | "
          f"window {WINDOW_MS}ms={WINDOW_SAMPLES}smp stride={STRIDE_SAMPLES} | "
          f"causal={CAUSAL} rectify={RECTIFY} env_lp={ENVELOPE_LP_HZ}Hz gyro_lp={GYRO_LP_HZ}Hz")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_windows = []
    all_labels  = []
    all_groups  = []
    per_subject_stats = {}

    for subject_id, files in SUBJECT_FILES.items():
        subj_pos = 0
        subj_neg = 0
        for path in files:
            if not path.exists():
                print(f"  [SKIP] Not found: {path}")
                continue
            print(f"  [{subject_id}] {path.name}")
            df = pd.read_csv(path)
            windows, labels = extract_from_circuit(df, subject_id)

            all_windows.extend(windows)
            all_labels.extend(labels)
            all_groups.extend([subject_id] * len(windows))

            p = sum(labels)
            n = len(labels) - p
            subj_pos += p
            subj_neg += n
            print(f"      +{p} intent / +{n} non-intent windows")

        per_subject_stats[subject_id] = {'positive': subj_pos, 'negative': subj_neg}

    if not all_windows:
        print("\nERROR: No windows extracted. Check SUBJECT_FILES paths.")
        return

    X = np.stack(all_windows, axis=0)
    y = np.array(all_labels, dtype=np.int8)
    groups = np.array(all_groups, dtype=object)

    print(f"\nBefore balancing: {np.sum(y==1)} intent, {np.sum(y==0)} non-intent")

    X, y, groups = balance_negatives(X, y, groups, MAX_NEG_TO_POS_RATIO, RANDOM_SEED)

    print(f"After balancing:  {np.sum(y==1)} intent, {np.sum(y==0)} non-intent")

    np.save(OUTPUT_DIR / 'inputs.npy', X)
    np.save(OUTPUT_DIR / 'labels.npy', y)
    np.save(OUTPUT_DIR / 'subject_ids.npy', groups)

    meta = {
        'config': {
            'RAW_RATE': RAW_RATE,
            'TARGET_RATE': TARGET_RATE,
            'DECIM': DECIM,
            'WINDOW_MS': WINDOW_MS,
            'WINDOW_SAMPLES': WINDOW_SAMPLES,
            'STRIDE_SAMPLES': STRIDE_SAMPLES,
            'CAUSAL': CAUSAL,
            'RECTIFY': RECTIFY,
            'FILTER_ORDER': FILTER_ORDER,
            'ENVELOPE_LP_HZ': ENVELOPE_LP_HZ,
            'GYRO_LP_HZ': GYRO_LP_HZ,
            'LATE_STANCE_FRACTION': LATE_STANCE_FRACTION,
            'PRE_TOEOFF_MARGIN_MS': PRE_TOEOFF_MARGIN_MS,
            'POSITIVE_OVERLAP_THRESHOLD': POSITIVE_OVERLAP_THRESHOLD,
            'NEGATIVE_OVERLAP_THRESHOLD': NEGATIVE_OVERLAP_THRESHOLD,
            'DISCARD_AMBIGUOUS': DISCARD_AMBIGUOUS,
            'MIN_STANCE_MS': MIN_STANCE_MS,
            'MAX_STANCE_MS': MAX_STANCE_MS,
            'MAX_NEG_TO_POS_RATIO': MAX_NEG_TO_POS_RATIO,
            'channels': ['TA_envelope_causal', 'Shank_Gy_causal'],
        },
        'summary': {
            'total_windows': int(len(y)),
            'intent_windows': int(np.sum(y == 1)),
            'non_intent_windows': int(np.sum(y == 0)),
            'X_shape': list(X.shape),
            'subjects': list(SUBJECT_FILES.keys()),
        },
        'per_subject': per_subject_stats,
    }
    with open(OUTPUT_DIR / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n" + "="*55)
    print("EXTRACTION COMPLETE (sliding window)")
    print("="*55)
    print(f"  Total windows:  {len(y)}")
    print(f"  Intent (1):     {np.sum(y==1)}")
    print(f"  Non-intent (0): {np.sum(y==0)}")
    print(f"  X shape:        {X.shape}")
    print(f"  Saved to:       {OUTPUT_DIR.resolve()}/")
    print(f"\n  Output files:")
    print(f"    inputs.npy       (N, 2, {WINDOW_SAMPLES})  -- ch0=TA_envelope, ch1=Shank_Gy")
    print(f"    labels.npy       (N,)               -- 1=intent, 0=no_intent")
    print(f"    subject_ids.npy  (N,)               -- subject ID per window")
    print(f"    meta.json")
    print(f"\n  Load in PyTorch:")
    print(f"    inputs      = np.load('{OUTPUT_DIR.name}/inputs.npy')")
    print(f"    labels      = np.load('{OUTPUT_DIR.name}/labels.npy')")
    print(f"    subject_ids = np.load('{OUTPUT_DIR.name}/subject_ids.npy', allow_pickle=True)")
    print(f"\n  Leave-one-subject-out example:")
    print(f"    val_mask   = (subject_ids == 'AB156')")
    print(f"    train_mask = ~val_mask")
    print(f"    train_X, train_y = inputs[train_mask], labels[train_mask]")
    print(f"    val_X,   val_y   = inputs[val_mask],   labels[val_mask]")


if __name__ == "__main__":
    main()
