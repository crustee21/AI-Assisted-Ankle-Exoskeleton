import time
import numpy as np

def _window(seconds, read_emg_raw, read_fsrs, read_imu_frame, live=True):
    emg, imu, fsr = [], [], []
    t0 = time.perf_counter(); last_print = 0
    while time.perf_counter() - t0 < seconds:
        emg.append(read_emg_raw()[1])
        if len(emg) % 10 == 0:
            imu.append(read_imu_frame()); fsr.append(read_fsrs())
        now = time.perf_counter() - t0
        if live and now - last_print >= 0.5 and imu:
            ax, ay, az, gx, gy, gz = imu[-1]; h, t = fsr[-1]
            e = np.array(emg[-200:])
            print(f"  {now:4.1f}s  EMG {e.mean():.3f}V ±{e.std():.3f}  "
                  f"gy {gy:+7.1f} dps  az {az:+5.2f} g  heel {h:.2f} V  toe {t:.2f} V")
            last_print = now
    return np.array(emg), np.array(imu), np.array(fsr)

def sensor_check(read_emg_raw, read_fsrs, read_imu_frame):
    try:
        verify_bmi160_scale()                      # from pynq_data_logger.py
    except NameError:
        print("(%run pynq_data_logger.py first)")
    ok = []

    print("\n[1/5] sit still relaxed  5 s")
    emg_r, imu_r, fsr_r = _window(5, read_emg_raw, read_fsrs, read_imu_frame)
    print("\n[2/5] pull ur toes up")
    emg_a, _, _ = _window(3, read_emg_raw, read_fsrs, read_imu_frame)
    print("\n[3/5] tilt IMU a few times")
    _, imu_m, _ = _window(3, read_emg_raw, read_fsrs, read_imu_frame)
    print("\n[4/5] press heel down")
    _, _, fsr_h = _window(3, read_emg_raw, read_fsrs, read_imu_frame)
    print("\n[5/5] press toes down")
    _, _, fsr_t = _window(3, read_emg_raw, read_fsrs, read_imu_frame)

    print("\nresult")
    dc = emg_r.mean(); rs, as_ = emg_r.std(), emg_a.std()
    clip = (emg_a.min() < 0.05) or (emg_a.max() > 3.25)
    good = 1.5 < dc < 2.4 and as_ > 1.5 * rs and not clip
    ok.append(good)
    print(f"EMG   DC {dc:.3f} V  rest std {rs:.4f}  active std {as_:.4f}  (x{as_ / rs:.1f})"
          f"{'  CLIPPING' if clip else ''}  -> {'OK' if good else 'CHECK: electrodes / gain / reference'}")
    # IMU: rest |a| ~ 1 g, gyro quiet at rest, gy moves when tilted
    a_rest = np.linalg.norm(imu_r[:, :3], axis=1).mean()
    gy_rest, gy_move = imu_r[:, 4].std(), np.abs(imu_m[:, 4]).max()
    good = 0.9 < a_rest < 1.1 and gy_rest < 5 and gy_move > 50
    ok.append(good)
    print(f"IMU   |a| at rest {a_rest:.2f} g  gy rest std {gy_rest:.1f} dps  gy peak when tilted {gy_move:.0f} dps"
          f"  -> {'OK' if good else 'CHECK: mount / axis / range regs'}")
    # FSRs: HIGH = loaded. Pressed level clearly above lifted level, each on its own channel.
    for name, col, d in (("heel", 0, fsr_h), ("toe", 1, fsr_t)):
        hi, lo = np.percentile(d[:, col], 95), np.percentile(d[:, col], 5)
        other = np.percentile(d[:, 1 - col], 95) - np.percentile(d[:, 1 - col], 5)
        good = hi - lo > 0.8 and other < (hi - lo) * 0.7
        ok.append(good)
        print(f"FSR   {name}: lifted {lo:.2f} V -> pressed {hi:.2f} V (swing {hi - lo:.2f} V; other channel moved {other:.2f} V)"
              f"  -> {'OK' if good else 'CHECK: insole placement / channels swapped'}")
    print("\nALL OK" if all(ok) else "\nSomething to fix before recording.")
    return all(ok)
