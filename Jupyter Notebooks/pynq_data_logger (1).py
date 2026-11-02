"""First-pass, non-actuating PYNQ-Z2 data recorder for the ankle project.

Run this file in the SAME Jupyter kernel that already contains the working
`mmio`, `write_reg`, `read_regs`, and `read_emg_raw` functions. It writes
timestamped raw CSVs; preprocessing and gait-event labels are deliberately
kept separate. EMG is paced at 1000 Hz; FSRs are a separate 100 Hz stream.
Every missed EMG deadline is recorded so inadequate timing is detected, not
hidden.

IMU CHANGE (2026): the BMI160 is now read DIRECTLY (12 bytes from 0x0C) at a
modest rate instead of draining the 1600 Hz FIFO. A small fixed-size read does
not block the 1000 Hz EMG loop the way a ~170-byte FIFO drain did. Set the
BMI160 ODR to 100 Hz (ACC/GYR_CONF = 0x28) before recording.
"""

import csv
import json
import time
from pathlib import Path
from datetime import datetime, timezone


BMI160 = 0x68
IMU_RATE_HZ = 100.0            # BMI160 ODR; set ACC/GYR_CONF = 0x28 in the notebook
IMU_PERIOD_S = 1.0 / IMU_RATE_HZ


def _signed16(lo, hi):
    value = (hi << 8) | lo
    return value - 65536 if value >= 32768 else value


def read_imu_frame():
    """Read the latest accel+gyro sample directly from the data registers.

    12 bytes from 0x0C: gyro x/y/z (0x0C-0x11) then accel x/y/z (0x12-0x17).
    No FIFO, so nothing to overflow and nothing to misalign. The timestamp is
    simply the read time, which is correct because these registers always hold
    the most recent sample.
    """
    d = read_regs(BMI160, 0x0C, 12)
    gx = _signed16(d[0], d[1])
    gy = _signed16(d[2], d[3])
    gz = _signed16(d[4], d[5])
    ax = _signed16(d[6], d[7])
    ay = _signed16(d[8], d[9])
    az = _signed16(d[10], d[11])
    return (ax / 16384.0, ay / 16384.0, az / 16384.0,
            gx / 131.2, gy / 131.2, gz / 131.2)


def start_imu_fifo():
    """Make sure the FIFO is OFF; this recorder reads data registers directly."""
    write_reg(BMI160, 0x47, 0x00)  # FIFO disabled
    time.sleep(0.005)


def verify_bmi160_scale():
    """Print the range registers used by the CSV conversion factors."""
    accel_range = read_regs(BMI160, 0x41, 1)[0]
    gyro_range = read_regs(BMI160, 0x43, 1)[0]
    print(f"ACC_RANGE = {hex(accel_range)} (expected 0x03 = +/-2 g)")
    print(f"GYR_RANGE = {hex(gyro_range)} (expected 0x03 = +/-250 dps)")
    if accel_range != 0x03 or gyro_range != 0x03:
        raise RuntimeError("BMI160 range differs from this logger's scale factors.")


def _timing_summary(times, missed):
    intervals = [b - a for a, b in zip(times, times[1:])]
    return {
        "sample_count": len(times),
        "mean_rate_hz": 1.0 / (sum(intervals) / len(intervals)) if intervals else 0.0,
        "max_interval_ms": max(intervals) * 1000 if intervals else 0.0,
        "missed_deadlines": missed,
    }


def benchmark_full_path(read_fsrs, duration_s=3.0, emg_rate_hz=1000.0,
                        fsr_rate_hz=100.0, imu_rate_hz=IMU_RATE_HZ):
    """Benchmark the exact EMG + 100 Hz FSR + direct-read IMU workload.

    This does not write files. The returned EMG timing is the pass/fail
    measurement for the 20--350 Hz EMG pipeline; IMU frame count should be
    close to imu_rate_hz * duration_s.
    """
    emg_period = 1.0 / emg_rate_hz
    fsr_period = 1.0 / fsr_rate_hz
    imu_period = 1.0 / imu_rate_hz
    start_imu_fifo()
    time.sleep(0.003)
    t0 = time.perf_counter()
    end = t0 + duration_s
    next_emg, next_fsr, next_imu = t0, t0, t0
    emg_times, emg_missed = [], 0
    fsr_count, imu_frame_count = 0, 0

    while time.perf_counter() < end:
        now = time.perf_counter()
        if now >= next_emg:
            late = max(0, int((now - next_emg) / emg_period))
            emg_missed += late
            read_emg_raw()
            emg_times.append(now)
            next_emg += (late + 1) * emg_period

        if now >= next_fsr:
            read_fsrs()
            fsr_count += 1
            next_fsr += fsr_period
            while next_fsr <= now:
                next_fsr += fsr_period

        if now >= next_imu:
            read_imu_frame()
            imu_frame_count += 1
            next_imu += imu_period
            while next_imu <= now:
                next_imu += imu_period

        now = time.perf_counter()
        if now < next_emg:
            time.sleep(min(next_emg - now, 0.0005))

    summary = _timing_summary(emg_times, emg_missed)
    summary.update({
        "fsr_sample_count": fsr_count,
        "imu_frame_count": imu_frame_count,
        "imu_expected_frames": round(imu_rate_hz * duration_s),
    })
    print(summary)
    return summary


def record_trial(activity, duration_s, read_fsrs, output_dir="ankle_dataset",
                 subject_id="subject01", emg_rate_hz=1000.0, fsr_rate_hz=100.0,
                 imu_rate_hz=IMU_RATE_HZ):
    """Record one fixed-activity trial without commanding the motor.

    Parameters
    ----------
    activity : str
        One stable activity name for this entire file, e.g. 'walk_level'.
    duration_s : float
        Trial length in seconds.
    read_fsrs : callable
        Your already-verified FSR function. It must return (heel, toe), as raw
        ADC values or volts; the metadata records no assumed units.

    Outputs
    -------
    <trial>_emg.csv  : raw EMG, paced at 1000 Hz, with a timestamp per sample
    <trial>_fsr.csv  : heel/toe FSRs, sampled independently at 100 Hz
    <trial>_imu.csv  : accel+gyro, read directly at imu_rate_hz
    <trial>_meta.json: activity label, requested duration and timing statistics
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{subject_id}_{activity}_{stamp}"
    emg_path = out / f"{prefix}_emg.csv"
    fsr_path = out / f"{prefix}_fsr.csv"
    imu_path = out / f"{prefix}_imu.csv"
    meta_path = out / f"{prefix}_meta.json"

    emg_period_s = 1.0 / emg_rate_hz
    fsr_period_s = 1.0 / fsr_rate_hz
    imu_period_s = 1.0 / imu_rate_hz

    start_imu_fifo()
    time.sleep(0.003)

    print(f"Starting {duration_s:g} s '{activity}' trial in 3 seconds...")
    time.sleep(3.0)
    print("RECORDING")

    t0 = time.perf_counter()
    end = t0 + float(duration_s)
    next_emg_sample, next_fsr_sample, next_imu = t0, t0, t0
    emg_times, emg_missed_deadlines, fsr_count = [], 0, 0
    imu_frames = 0

    with (emg_path.open("w", newline="") as emg_file,
          fsr_path.open("w", newline="") as fsr_file,
          imu_path.open("w", newline="") as imu_file):
        emg_writer = csv.writer(emg_file)
        fsr_writer = csv.writer(fsr_file)
        imu_writer = csv.writer(imu_file)
        emg_writer.writerow(["t_s", "emg_counts", "emg_volts"])
        fsr_writer.writerow(["t_s", "fsr_heel", "fsr_toe"])
        imu_writer.writerow(["t_s", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"])

        while time.perf_counter() < end:
            now = time.perf_counter()
            if now >= next_emg_sample:
                missed_now = max(0, int((now - next_emg_sample) / emg_period_s))
                emg_missed_deadlines += missed_now
                emg_counts, emg_volts = read_emg_raw()
                emg_writer.writerow([now - t0, emg_counts, emg_volts])
                emg_times.append(now)
                next_emg_sample += (missed_now + 1) * emg_period_s

            if now >= next_fsr_sample:
                heel, toe = read_fsrs()
                fsr_writer.writerow([now - t0, heel, toe])
                fsr_count += 1
                next_fsr_sample += fsr_period_s
                while next_fsr_sample <= now:
                    next_fsr_sample += fsr_period_s

            if now >= next_imu:
                ax, ay, az, gx, gy, gz = read_imu_frame()
                imu_writer.writerow([now - t0, ax, ay, az, gx, gy, gz])
                imu_frames += 1
                next_imu += imu_period_s
                while next_imu <= now:
                    next_imu += imu_period_s

            # Yield only until the next 1 kHz EMG deadline; this avoids a
            # CPU-burning busy wait while preserving the paced acquisition rate.
            now = time.perf_counter()
            if now < next_emg_sample:
                time.sleep(min(next_emg_sample - now, 0.0005))

        emg_file.flush()
        fsr_file.flush()
        imu_file.flush()

    emg_timing = _timing_summary(emg_times, emg_missed_deadlines)
    metadata = {
        "subject_id": subject_id,
        "activity": activity,
        "requested_duration_s": float(duration_s),
        "emg_sample_count": emg_timing["sample_count"],
        "emg_target_rate_hz": emg_rate_hz,
        "emg_mean_rate_hz": emg_timing["mean_rate_hz"],
        "emg_max_interval_ms": emg_timing["max_interval_ms"],
        "emg_missed_deadlines": emg_timing["missed_deadlines"],
        "fsr_sample_count": fsr_count,
        "fsr_target_rate_hz": fsr_rate_hz,
        "imu_frame_count": imu_frames,
        "imu_target_rate_hz": imu_rate_hz,
        "fsr_units": "raw values supplied by read_fsrs",
        "notes": "Raw collection only. Add heel-contact/toe-off intent labels later from FSR/video alignment.",
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print("DONE")
    print(f"EMG samples: {metadata['emg_sample_count']} ({metadata['emg_mean_rate_hz']:.1f} Hz)")
    print(f"FSR samples: {metadata['fsr_sample_count']}")
    print(f"IMU frames:   {imu_frames} ({imu_frames / float(duration_s):.1f} Hz)")
    print(f"Saved: {emg_path}, {fsr_path}, {imu_path}, {meta_path}")
    return emg_path, fsr_path, imu_path, meta_path


def record_standing_baseline(read_fsrs, duration_s=30, **kwargs):
    """Record the standing-still baseline required for session z-scoring."""
    return record_trial("standing_still", duration_s, read_fsrs, **kwargs)