import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, lfilter

import extract_dataset_sliding as ex

SUBJECT_ID          = "REMI"
BASELINE_ACTIVITY   = "standing_still"
WALK_ACTIVITY_RE    = r"^walk"

EMG_COL             = "emg_volts"
EMG_BP_HZ           = (20.0, 350.0)
EMG_BP_ORDER        = 6
EMG_NOTCH_HZ        = (60.0, 180.0, 300.0)
EMG_NOTCH_Q         = 30.0

GYRO_COL            = "gy_dps"
GYRO_SIGN           = +1.0
GYRO_STD_FLOOR_DPS  = 10.0

FSR_HI_PCT, FSR_LO_PCT = 95, 5
FSR_ON_FRAC         = 0.55
FSR_OFF_FRAC        = 0.45
MIN_EVENT_GAP_MS    = 250

STOMP_SEARCH_S      = 12.0
STOMP_ACC_G         = 0.6
STOMP_MARGIN_S      = 1.0
BASELINE_SKIP_S     = 1.0

CLIP_ABS            = 31.99
MAX_STANCE_MS       = 2500

FNAME_RE = re.compile(r"^(?P<subject>[^_]+)_(?P<activity>.+)_(?P<stamp>\d{8}T\d{6}Z)_(?P<stream>emg|fsr|imu)\.csv$")

def find_trials(data_dir):
    trials = {}
    for p in sorted(Path(data_dir).glob("*.csv")):
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        key = f"{m['activity']}_{m['stamp']}"
        trials.setdefault(key, {"activity": m["activity"]})[m["stream"]] = p
    return {k: v for k, v in trials.items() if all(s in v for s in ("emg", "fsr", "imu"))}

def load_trial(t):
    emg = pd.read_csv(t["emg"]); fsr = pd.read_csv(t["fsr"]); imu = pd.read_csv(t["imu"])
    t_end = min(emg.t_s.iloc[-1], fsr.t_s.iloc[-1], imu.t_s.iloc[-1])
    grid = np.arange(0.0, t_end, 1.0 / ex.RAW_RATE)
    return {
        "t":    grid,
        "emg":  np.interp(grid, emg.t_s, emg[EMG_COL]),
        "gyro": np.interp(grid, imu.t_s, GYRO_SIGN * imu[GYRO_COL]),
        "acc":  np.stack([np.interp(grid, imu.t_s, imu[c]) for c in ("ax_g", "ay_g", "az_g")], 1),
        "heel": np.interp(grid, fsr.t_s, fsr.fsr_heel),
        "toe":  np.interp(grid, fsr.t_s, fsr.fsr_toe),
        "emg_rate": len(emg) / emg.t_s.iloc[-1],
        "emg_max_gap_ms": emg.t_s.diff().max() * 1000,
    }

def emg_precondition(x, fs=ex.RAW_RATE):
    b, a = butter(EMG_BP_ORDER, [EMG_BP_HZ[0] / (fs / 2), EMG_BP_HZ[1] / (fs / 2)], btype="band")
    x = lfilter(b, a, x - np.mean(x))
    for f0 in EMG_NOTCH_HZ:
        b, a = iirnotch(f0, EMG_NOTCH_Q, fs)
        x = lfilter(b, a, x)
    return x

def condition(rec):
    env = ex.emg_envelope(emg_precondition(rec["emg"]))
    gy  = ex.gyro_conditioned(rec["gyro"])
    sl  = slice(None, None, ex.DECIM)
    return env[sl], gy[sl]

def stomp_bounds(rec):
    n = len(rec["t"]); fs = ex.RAW_RATE
    dev = np.abs(np.linalg.norm(rec["acc"], axis=1) - 1.0)
    hit = np.where(dev > STOMP_ACC_G)[0]
    w = int(STOMP_SEARCH_S * fs); m = int(STOMP_MARGIN_S * fs)
    head = hit[hit < w]; tail = hit[hit > n - w]
    start = head[-1] + m if len(head) else 0
    end   = tail[0]  - m if len(tail) else n
    return max(0, start), min(n, end), len(head) > 0, len(tail) > 0

def edges(sig, rising):
    lo, hi = np.percentile(sig, [FSR_LO_PCT, FSR_HI_PCT])
    on  = lo + FSR_ON_FRAC  * (hi - lo)
    off = lo + FSR_OFF_FRAC * (hi - lo)
    loaded = sig[0] > on
    out, last = [], -10**9
    gap = int(MIN_EVENT_GAP_MS * ex.RAW_RATE / 1000)
    for i, v in enumerate(sig):
        if not loaded and v > on:
            loaded = True
            if rising and i - last > gap: out.append(i); last = i
        elif loaded and v < off:
            loaded = False
            if not rising and i - last > gap: out.append(i); last = i
    return np.array(out, dtype=int)

def event_frame(rec, start, end, mode):
    n = end - start
    df = pd.DataFrame({"Mode": np.full(n, mode, dtype=int),
                       "Heel_Contact": np.zeros(n, dtype=int),
                       "Toe_Off": np.zeros(n, dtype=int)})
    hc = to = np.array([], dtype=int)
    if mode == ex.MODE_WALKING:
        hc = edges(rec["heel"][start:end], rising=True)
        to = edges(rec["toe"][start:end],  rising=False)
        df.loc[hc, "Heel_Contact"] = hc
        df.loc[to, "Toe_Off"] = to
    return df, hc, to

def slide(ta_env, gy, mode, intent_mask):
    W, S = ex.WINDOW_SAMPLES, ex.STRIDE_SAMPLES
    windows, labels, starts = [], [], []
    start = 0
    while start + W <= len(mode):
        end = start + W
        wm = mode[start:end]
        if wm[0] in (ex.MODE_WALKING, ex.MODE_STANDING) and np.all(wm == wm[0]):
            ov = np.sum(intent_mask[start:end]) / W
            label = None
            if ov >= ex.POSITIVE_OVERLAP_THRESHOLD: label = 1
            elif ov <= ex.NEGATIVE_OVERLAP_THRESHOLD: label = 0
            elif not ex.DISCARD_AMBIGUOUS: label = 1 if ov >= 0.5 else 0
            if label is not None:
                windows.append(np.stack([ta_env[start:end], gy[start:end]], 0))
                labels.append(label)
                starts.append(start / ex.TARGET_RATE)
        start += S
    return windows, labels, starts

def finish(x):
    x = x.astype(np.float32)
    return np.clip(x, -CLIP_ABS, CLIP_ABS) if CLIP_ABS else x

def main():
    ap = argparse.ArgumentParser(description="Extract own logger recordings into the sliding-window dataset format.")
    ap.add_argument("--data-dir", required=True, help="folder with the *_emg/_fsr/_imu.csv triplets")
    ap.add_argument("--out", default="own_dataset_sliding")
    ap.add_argument("--subject", default=SUBJECT_ID)
    ap.add_argument("--env-cutoff", type=float, default=40.0, help="EMG envelope LP (Hz); must match the checkpoint (40)")
    ap.add_argument("--gyro-cutoff", type=float, default=ex.GYRO_LP_HZ)
    ap.add_argument("--plot", action="store_true", help="write per-trial event overlay PNGs into --out")
    args = ap.parse_args()

    ex.ENVELOPE_LP_HZ = args.env_cutoff
    ex.GYRO_LP_HZ     = args.gyro_cutoff
    ex.MAX_STANCE_MS  = MAX_STANCE_MS
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    trials = find_trials(args.data_dir)
    base_keys = [k for k, v in trials.items() if v["activity"] == BASELINE_ACTIVITY]
    walk_keys = [k for k, v in trials.items() if re.match(WALK_ACTIVITY_RE, v["activity"])]
    if not base_keys:
        raise SystemExit(f"No '{BASELINE_ACTIVITY}' recording found in {args.data_dir}")
    print(f"Found {len(walk_keys)} walking trial(s), {len(base_keys)} baseline(s)")
    print(f"Chain: BP {EMG_BP_HZ} + notch {EMG_NOTCH_HZ} -> rectify -> LP {ex.ENVELOPE_LP_HZ} Hz | "
          f"gyro {GYRO_COL} x{GYRO_SIGN:+.0f} -> LP {ex.GYRO_LP_HZ} Hz | decim {ex.DECIM} | clip +/-{CLIP_ABS}")

    base = load_trial(trials[base_keys[0]])
    b_env, b_gy = condition(base)
    skip = int(BASELINE_SKIP_S * ex.TARGET_RATE)
    stats = {"env": (float(b_env[skip:].mean()), float(b_env[skip:].std())),
             "gy":  (float(b_gy[skip:].mean()),  max(float(b_gy[skip:].std()), GYRO_STD_FLOOR_DPS))}
    print(f"Baseline [{base_keys[0]}]: env mu={stats['env'][0]:.4g} std={stats['env'][1]:.4g} | "
          f"gyro mu={stats['gy'][0]:.3g} std={stats['gy'][1]:.3g} dps (raw std {b_gy[skip:].std():.3g}, floor {GYRO_STD_FLOOR_DPS})")

    X, y, trial_ids, t0, report = [], [], [], [], {}

    mode = np.full(len(b_env), ex.MODE_STANDING)
    w, l, st = slide(finish(ex.zscore(b_env, *stats["env"])), finish(ex.zscore(b_gy, *stats["gy"])),
                     mode[skip:], np.zeros(len(mode), bool))
    X += w; y += l; trial_ids += [base_keys[0]] * len(w); t0 += st
    report[base_keys[0]] = {"activity": BASELINE_ACTIVITY, "windows": len(w), "positive": 0}

    for k in walk_keys:
        rec = load_trial(trials[k])
        s0, s1, had_head, had_tail = stomp_bounds(rec)
        env, gy = condition(rec)
        df, hc, to = event_frame(rec, s0, s1, ex.MODE_WALKING)
        mask_raw = ex.build_intent_mask(df, "Toe_Off", "Heel_Contact")

        sl = slice(s0 // ex.DECIM, s1 // ex.DECIM)
        env_z = finish(ex.zscore(env[sl], *stats["env"]))
        gy_z  = finish(ex.zscore(gy[sl],  *stats["gy"]))
        mask  = mask_raw[::ex.DECIM][: len(env_z)]
        mode  = np.full(len(env_z), ex.MODE_WALKING)
        w, l, st = slide(env_z, gy_z, mode, mask)

        stances = []
        for t_ in to:
            prev = hc[hc < t_]
            if len(prev): stances.append(t_ - prev[-1])
        stances = np.array(stances)
        kept = int(((stances >= ex.MIN_STANCE_MS) & (stances <= ex.MAX_STANCE_MS)).sum())
        X += w; y += l; trial_ids += [k] * len(w); t0 += st
        report[k] = {
            "activity": trials[k]["activity"], "windows": len(w), "positive": int(sum(l)),
            "emg_rate_hz": round(rec["emg_rate"], 1), "emg_max_gap_ms": round(rec["emg_max_gap_ms"], 1),
            "kept_s": [round(s0 / ex.RAW_RATE, 2), round(s1 / ex.RAW_RATE, 2)],
            "stomps_found": [had_head, had_tail],
            "heel_contacts": int(len(hc)), "toe_offs": int(len(to)), "valid_strides": kept,
            "stance_ms": {"median": float(np.median(stances)) if len(stances) else None,
                          "min": float(stances.min()) if len(stances) else None,
                          "max": float(stances.max()) if len(stances) else None},
            "env_z_p99": float(np.percentile(env_z, 99)), "gyro_z_absmax": float(np.abs(gy_z).max()),
        }
        r = report[k]
        print(f"  [{k}] kept {r['kept_s'][0]}-{r['kept_s'][1]}s | HC {len(hc)} TO {len(to)} "
              f"valid strides {kept} | stance med {r['stance_ms']['median']} ms | "
              f"+{r['positive']} / -{len(l) - r['positive']} windows")
        if not (had_head and had_tail):
            print(f"      WARNING: sync stomps not found at {'start' if not had_head else ''}"
                  f"{' end' if not had_tail else ''} - nothing trimmed there")
        if args.plot:
            plot_trial(out / f"{k}_events.png", rec, s0, s1, hc, to, mask_raw, env_z, gy_z)

    if not X:
        raise SystemExit("No windows extracted.")
    X = np.stack(X); y = np.array(y, dtype=np.int8)
    trial_ids = np.array(trial_ids, dtype=object)
    t0 = np.array(t0, dtype=np.float32)
    subj = np.array([args.subject] * len(y), dtype=object)

    print(f"\nBefore balancing: {int((y == 1).sum())} intent, {int((y == 0).sum())} non-intent")
    keep = np.arange(len(y))
    X, y, keep = ex.balance_negatives(X, y, keep, ex.MAX_NEG_TO_POS_RATIO, ex.RANDOM_SEED)
    trial_ids, subj, t0 = trial_ids[keep], subj[keep], t0[keep]
    print(f"After balancing:  {int((y == 1).sum())} intent, {int((y == 0).sum())} non-intent")

    g1 = X[y == 1, 1, :].mean(0)
    print(f"Class-1 gyro mean across window: first {g1[0]:+.2f} -> last {g1[-1]:+.2f}  "
          f"(ENABL3S: +0.06 -> +0.70). {'OK' if g1[-1] > g1[0] and g1[-1] > 0 else 'CHECK GYRO_SIGN'}")
    e1 = X[y == 1, 0, :].mean(0); e0 = X[y == 0, 0, :].mean(0)
    print(f"EMG env mean: class-1 {e1.mean():+.2f}  class-0 {e0.mean():+.2f}  (ENABL3S: -0.37 / +0.17)")

    np.save(out / "inputs.npy", X)
    np.save(out / "labels.npy", y)
    np.save(out / "subject_ids.npy", subj)
    np.save(out / "trial_ids.npy", trial_ids)
    np.save(out / "window_t0.npy", t0)
    meta = {
        "config": {**{k: getattr(ex, k) for k in (
            "RAW_RATE", "TARGET_RATE", "DECIM", "WINDOW_MS", "WINDOW_SAMPLES", "STRIDE_SAMPLES",
            "CAUSAL", "RECTIFY", "FILTER_ORDER", "ENVELOPE_LP_HZ", "GYRO_LP_HZ",
            "LATE_STANCE_FRACTION", "PRE_TOEOFF_MARGIN_MS", "POSITIVE_OVERLAP_THRESHOLD",
            "NEGATIVE_OVERLAP_THRESHOLD", "DISCARD_AMBIGUOUS", "MIN_STANCE_MS", "MAX_STANCE_MS",
            "MAX_NEG_TO_POS_RATIO")},
            "EMG_BP_HZ": EMG_BP_HZ, "EMG_BP_ORDER": EMG_BP_ORDER, "EMG_NOTCH_HZ": EMG_NOTCH_HZ,
            "GYRO_COL": GYRO_COL, "GYRO_SIGN": GYRO_SIGN, "GYRO_STD_FLOOR_DPS": GYRO_STD_FLOOR_DPS, "CLIP_ABS": CLIP_ABS,
            "FSR_ON_FRAC": FSR_ON_FRAC, "FSR_OFF_FRAC": FSR_OFF_FRAC,
            "baseline_trial": base_keys[0], "baseline_stats": stats,
            "channels": ["TA_envelope_causal", "Shank_Gy_causal"]},
        "summary": {"total_windows": int(len(y)), "intent_windows": int((y == 1).sum()),
                    "non_intent_windows": int((y == 0).sum()), "X_shape": list(X.shape),
                    "subjects": [args.subject], "trials": list(report)},
        "per_trial": report,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {X.shape} -> {out.resolve()}/  (inputs, labels, subject_ids, trial_ids, window_t0, meta.json)")

def plot_trial(path, rec, s0, s1, hc, to, mask_raw, env_z, gy_z):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = rec["t"]; fs = ex.RAW_RATE
    fig, ax = plt.subplots(3, 1, figsize=(18, 9), sharex=True)
    ax[0].plot(t, rec["heel"], label="heel FSR"); ax[0].plot(t, rec["toe"], label="toe FSR")
    for i in hc: ax[0].axvline(t[s0 + i], color="g", lw=0.8)
    for i in to: ax[0].axvline(t[s0 + i], color="r", lw=0.8)
    ax[0].axvspan(t[0], t[s0], color="k", alpha=0.15); ax[0].axvspan(t[s1 - 1], t[-1], color="k", alpha=0.15)
    ax[0].legend(loc="lower right"); ax[0].set_title("green = heel contact, red = toe-off, shaded = trimmed")
    tt = t[s0:s1:ex.DECIM][: len(env_z)]
    for a, sig, name in ((ax[1], env_z, "TA envelope (z)"), (ax[2], gy_z, "shank gyro (z)")):
        a.plot(tt, sig, lw=0.8); a.set_ylabel(name)
        a.fill_between(tt, sig.min(), sig.max(), where=mask_raw[::ex.DECIM][: len(sig)], color="orange", alpha=0.3)
    ax[2].set_xlabel("s  (orange = intent region)")
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close()

if __name__ == "__main__":
    main()
