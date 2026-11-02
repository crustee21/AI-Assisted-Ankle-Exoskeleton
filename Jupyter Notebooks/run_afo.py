"""
run_afo.py -- the closed loop: sensors -> preprocessing -> FPGA inference -> safety layer -> motor.

Like pynq_data_logger.py, this is meant to be `%run` in the Jupyter kernel that
already has `read_emg_raw`, `read_fsrs`, `read_regs`, `write_reg`. Then:

    ol = Overlay("footdrop.bit")
    fpga = FpgaInference(ol)
    calib = calibrate(read_emg_raw, read_fsrs, read_imu_frame)     # 5 s standing, 5 s foot lifted
    motor = NoMotor()                                               # until the bench test passes
    run(fpga, calib, motor, read_emg_raw, read_fsrs, read_imu_frame, duration_s=60, log="walk_live_01.csv")

Every 10 ms tick:
    EMG block since last tick -> causal BP/notch/rectify/LP (same filters as the extractor,
    state carried between blocks so it is identical to filtering the stream) -> last sample
    gyro -> causal 30 Hz LP -> z-score (std floored like the extractor)
    one {gyro, emg} sample into the FPGA, pulse ap_start, read logit + valid bit
    SafetyLayer.step() -> torque fraction -> motor.command(fraction * TORQUE_FULL_NM)
    one CSV row

The log is the input to replay_safety.py on the PC once you add real logits.
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi
from pynq import MMIO

from safety_layer import SafetyLayer, SafetyConfig
from can_mcp import NoMotor

# --- must match extract_own_data.py / extract_dataset_sliding.py ----------------------
EMG_FS            = 1000.0
TICK_HZ           = 100.0
EMG_BP_HZ, EMG_BP_ORDER = (20.0, 350.0), 6
EMG_NOTCH_HZ, EMG_NOTCH_Q = (60.0, 180.0, 300.0), 30.0
ENV_LP_HZ, ENV_ORDER = 40.0, 4            # extractor: ENVELOPE_LP_HZ / FILTER_ORDER
GYRO_LP_HZ, GYRO_ORDER = 30.0, 4
GYRO_STD_FLOOR_DPS = 10.0
CLIP_ABS           = 31.99                # ap_fixed<16,6>
SCALE              = 2 ** 10
GYRO_AXIS          = 4                    # index of gy_dps in read_imu_frame()'s tuple
TORQUE_FULL_NM     = 2.0                  # what torque=1.0 from the safety layer means; raise deliberately

# ---------------------------------------------------------------------------------------
class Causal:
    """One IIR section with carried state (sample-by-sample identical to streaming lfilter)."""
    def __init__(self, b, a):
        self.b, self.a = b, a
        self.z = lfilter_zi(b, a) * 0.0
    def __call__(self, x):
        y, self.z = lfilter(self.b, self.a, np.atleast_1d(x), zi=self.z)
        return y

class Preprocessor:
    def __init__(self):
        nyq = EMG_FS / 2
        self.bp = Causal(*butter(EMG_BP_ORDER, [EMG_BP_HZ[0] / nyq, EMG_BP_HZ[1] / nyq], btype="band"))
        self.notches = [Causal(*iirnotch(f, EMG_NOTCH_Q, EMG_FS)) for f in EMG_NOTCH_HZ]
        self.env_lp = Causal(*butter(ENV_ORDER, ENV_LP_HZ / nyq))
        self.gy_lp = Causal(*butter(GYRO_ORDER, GYRO_LP_HZ / (TICK_HZ / 2)))   # gyro runs at tick rate
        self.emg_dc = None
    def emg_block(self, block_volts):
        """Process the EMG samples that arrived this tick; return the envelope at the tick."""
        x = np.asarray(block_volts, dtype=float)
        if self.emg_dc is None:
            self.emg_dc = x.mean()
        self.emg_dc += 0.001 * (x.mean() - self.emg_dc)     # slow DC tracker (mid-rail)
        y = self.bp(x - self.emg_dc)
        for n in self.notches:
            y = n(y)
        return float(self.env_lp(np.abs(y))[-1])
    def gyro(self, gy_dps):
        return float(self.gy_lp(gy_dps)[-1])

# ---------------------------------------------------------------------------------------
class FpgaInference:
    GPIO_DATA, GPIO2_DATA, LOGIT, LOGIT_CTRL = 0x00, 0x08, 0x10, 0x14
    def __init__(self, ol):
        g = ol.ip_dict["gpio_stim"]; c = ol.ip_dict["inference_stream_0"]
        self.gpio = MMIO(g["phys_addr"], g["addr_range"])
        self.ctrl = MMIO(c["phys_addr"], c["addr_range"])
    @staticmethod
    def _fx(v):
        q = int(round(max(-CLIP_ABS, min(CLIP_ABS, v)) * SCALE))
        return max(-32768, min(32767, q)) & 0xFFFF
    def push(self, emg_z, gy_z, timeout_s=0.008):
        """One sample in, one logit out. Returns (logit, valid)."""
        self.gpio.write(self.GPIO_DATA, (self._fx(gy_z) << 16) | self._fx(emg_z))
        self.gpio.write(self.GPIO2_DATA, 1)
        self.gpio.write(self.GPIO2_DATA, 0)
        t0 = time.perf_counter()
        while not (self.ctrl.read(self.LOGIT_CTRL) & 1):
            if time.perf_counter() - t0 > timeout_s:
                return 0.0, False
        q = self.ctrl.read(self.LOGIT) & 0xFFFF
        return (q - 0x10000 if q & 0x8000 else q) / SCALE, True

# ---------------------------------------------------------------------------------------
def _acquire(seconds, read_emg_raw, read_fsrs, read_imu_frame, pre):
    """Run the sensor loop for `seconds` without the FPGA; return per-tick arrays."""
    env, gy, heel, toe = [], [], [], []
    t0 = time.perf_counter(); next_tick = t0; block = []
    while time.perf_counter() < t0 + seconds:
        block.append(read_emg_raw()[1])
        now = time.perf_counter()
        if now >= next_tick:
            env.append(pre.emg_block(block)); block = []
            gy.append(pre.gyro(read_imu_frame()[GYRO_AXIS]))
            h, t = read_fsrs(); heel.append(h); toe.append(t)
            next_tick += 1 / TICK_HZ
            while next_tick <= now: next_tick += 1 / TICK_HZ
    return map(np.array, (env, gy, heel, toe))

def calibrate(read_emg_raw, read_fsrs, read_imu_frame, path="calibration.json", seconds=5.0):
    pre = Preprocessor()
    input(f"Stand still, relaxed, weight slightly forward. Enter to record {seconds:.0f} s...")
    _acquire(1.0, read_emg_raw, read_fsrs, read_imu_frame, pre)          # filter settle
    env, gy, heel, toe = _acquire(seconds, read_emg_raw, read_fsrs, read_imu_frame, pre)
    cal = {"env_mu": float(env.mean()), "env_std": float(env.std()),
           "gy_mu": float(gy.mean()), "gy_std": max(float(gy.std()), GYRO_STD_FLOOR_DPS),
           "gy_std_raw": float(gy.std()),
           "heel_loaded": float(np.median(heel)), "toe_loaded": float(np.median(toe))}
    input(f"Now lift the foot clear of the ground. Enter to record {seconds:.0f} s...")
    _, _, heel, toe = _acquire(seconds, read_emg_raw, read_fsrs, read_imu_frame, pre)
    cal["heel_unloaded"], cal["toe_unloaded"] = float(np.median(heel)), float(np.median(toe))
    Path(path).write_text(json.dumps(cal, indent=2))
    print(json.dumps(cal, indent=2))
    if cal["heel_loaded"] - cal["heel_unloaded"] < 0.5 or cal["toe_loaded"] - cal["toe_unloaded"] < 0.5:
        print("WARNING: FSR loaded/unloaded separation < 0.5 V -- check insole placement")
    return cal

# ---------------------------------------------------------------------------------------
def run(fpga, cal, motor, read_emg_raw, read_fsrs, read_imu_frame,
        duration_s=60.0, log="afo_live.csv", cfg=None):
    pre = Preprocessor()
    layer = SafetyLayer(cfg or SafetyConfig())
    layer.calibrate(cal["heel_loaded"], cal["toe_loaded"], cal["heel_unloaded"], cal["toe_unloaded"])
    for _ in range(32):                      # flush the FPGA's sliding window
        fpga.push(0.0, 0.0)
    motor.enable()
    print(f"RUNNING {duration_s:g} s  (torque ceiling {motor.ceiling} N*m, full = {TORQUE_FULL_NM} N*m)")

    t0 = time.perf_counter(); next_tick = t0; block = []
    n_ticks = missed = 0; last_state = None
    try:
        with open(log, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_ms", "emg_env", "emg_z", "gy_dps", "gy_z", "heel", "toe",
                        "logit", "logit_valid", "state", "torque_frac", "torque_nm", "reason", "emg_n"])
            while time.perf_counter() < t0 + duration_s:
                block.append(read_emg_raw()[1])
                now = time.perf_counter()
                if now < next_tick:
                    continue
                t_ms = (now - t0) * 1000
                env = pre.emg_block(block); emg_n = len(block); block = []
                gy_raw = read_imu_frame()[GYRO_AXIS]
                gy = pre.gyro(gy_raw)
                heel, toe = read_fsrs()
                emg_z = (env - cal["env_mu"]) / cal["env_std"]
                gy_z = (gy - cal["gy_mu"]) / cal["gy_std"]
                logit, valid = fpga.push(emg_z, gy_z)
                cmd = layer.step(t_ms, heel, toe, gy, logit, valid)
                nm = motor.command(cmd.torque * TORQUE_FULL_NM)
                w.writerow([f"{t_ms:.1f}", f"{env:.5f}", f"{emg_z:.3f}", f"{gy_raw:.2f}", f"{gy_z:.3f}",
                            f"{heel:.3f}", f"{toe:.3f}", f"{logit:.3f}", int(valid),
                            cmd.state, cmd.torque, f"{nm:.3f}", cmd.reason, emg_n])
                if cmd.state != last_state:
                    print(f"{t_ms:8.0f} ms  {cmd.state:10s} {cmd.reason}")
                    last_state = cmd.state
                n_ticks += 1
                next_tick += 1 / TICK_HZ
                if next_tick <= now:
                    missed += int((now - next_tick) * TICK_HZ) + 1
                    while next_tick <= now: next_tick += 1 / TICK_HZ
    finally:
        motor.command(0.0)
        motor.disable()
        print(f"STOPPED. {n_ticks} ticks, {missed} missed ({100 * missed / max(n_ticks, 1):.1f}%). log: {log}")
