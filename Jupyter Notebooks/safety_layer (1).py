from dataclasses import dataclass, field

@dataclass
class SafetyConfig:
    tick_ms:          float = 10.0
    #FSR hysteresis
    fsr_on_frac:      float = 0.55    # loaded when > lo + frac*(hi-lo)
    fsr_off_frac:     float = 0.45    # unloaded when <
    toe_on_frac:      float = 0.25    #toe-only
    toe_off_frac:     float = 0.15
    fsr_range_margin: float = 0.15    #fraction outside [lo, hi] that counts as a bad reading
    #gate
    min_stance_ms:    float = 300.0   #positive logit this soon after heel strike is ignored
    logit_threshold:  float = 0.0
    #confirm
    #asking
    # was wrong
    max_spark_ms:     float = 1500.0  #ceiling on how long a spark can be held unconfirmed
    logit_drop_ms:    float = 400.0   #logit must stay below threshold this long to cancel
    gy_confirm_dps:   float = 120.0   # |shank gyro| above this confirms (live swing peaks ~175-185 dps)
    # veto
    max_assist_ms:    float = 900.0   # swing never lasts this long; release regardless
    max_stance_ms:    float = 3000.0  # both loaded this long = standing; block sparks
    # recovery
    refractory_ms:    float = 400.0
    cancel_refractory_ms: float = 200.0
    fault_ms:         float = 200.0   # bad reading must persist this long to declare a fault
    fault_clear_ms:   float = 500.0   # ...and be clean this long to clear it
    logit_stall_ticks:int   = 3       # missing valid bits before inference is considered dead
    # outputs (units are whatever your CAN mapping expects; fractions of max here)
    spark_torque:     float = 0.15
    assist_torque:    float = 1.0

@dataclass
class Command:
    torque: float
    state:  str
    reason: str = ""          # non-empty only on the tick a transition happened
    phase:  str = ""          # STANCE / SWING (from FSRs) for logging
    heel_loaded: bool = False
    toe_loaded:  bool = False


class _Hysteresis:
    """loaded/unloaded with two thresholds; reports rising/falling edges."""
    def __init__(self, lo, hi, on_frac, off_frac):
        self.on  = lo + on_frac  * (hi - lo)
        self.off = lo + off_frac * (hi - lo)
        self.loaded = True        # calibrated while standing on it
    def update(self, v):
        rise = fall = False
        if not self.loaded and v > self.on:
            self.loaded, rise = True, True
        elif self.loaded and v < self.off:
            self.loaded, fall = False, True
        return rise, fall


class SafetyLayer:
    def __init__(self, cfg: SafetyConfig = None):
        self.cfg = cfg or SafetyConfig()
        self.heel = self.toe = None
        self.state = "FAULT"          # nothing moves until calibrate() has run
        self.t_state = 0.0
        self.t_stance_start = 0.0
        self.t_bad_since = None
        self.t_good_since = None
        self.stall = 0
        self.range = None
        self.t_logit_low = None

    # -- calibration --------------------------------------------------------
    def calibrate(self, heel_loaded, toe_loaded, heel_unloaded, toe_unloaded):
        """Levels (volts) from two short captures: standing on the foot, foot lifted."""
        c = self.cfg
        self.heel = _Hysteresis(heel_unloaded, heel_loaded, c.fsr_on_frac, c.fsr_off_frac)
        self.toe  = _Hysteresis(toe_unloaded,  toe_loaded,
                               c.toe_on_frac  if c.toe_on_frac  is not None else c.fsr_on_frac,
                               c.toe_off_frac if c.toe_off_frac is not None else c.fsr_off_frac)
        self.range = {}
        for name, lo, hi in (("heel", heel_unloaded, heel_loaded), ("toe", toe_unloaded, toe_loaded)):
            m = c.fsr_range_margin * (hi - lo)
            self.range[name] = (lo - m, hi + m)
        self._go("IDLE", 0.0, "calibrated")

    # -- one tick -------------------------------------------------------------
    def step(self, t_ms, heel_v, toe_v, gy_dps, logit, logit_valid=True) -> Command:
        c = self.cfg
        if self.heel is None:
            return Command(0.0, "FAULT", "", "", False, False)

        # 1. FSR phase tracking (always runs, even in FAULT, so edges aren't lost)
        heel_rise, _ = self.heel.update(heel_v)
        _, toe_fall  = self.toe.update(toe_v)
        stance = self.heel.loaded or self.toe.loaded
        if heel_rise:
            self.t_stance_start = t_ms
        stance_ms = t_ms - self.t_stance_start if stance else 0.0
        heel_contact = heel_rise
        toe_off = toe_fall

        # 2. Fault detection
        bad = (not self.range["heel"][0] <= heel_v <= self.range["heel"][1]
               or not self.range["toe"][0] <= toe_v <= self.range["toe"][1])
        self.stall = 0 if logit_valid else self.stall + 1
        bad = bad or self.stall >= c.logit_stall_ticks
        if bad:
            self.t_good_since = None
            self.t_bad_since = self.t_bad_since if self.t_bad_since is not None else t_ms
        else:
            self.t_bad_since = None
            self.t_good_since = self.t_good_since if self.t_good_since is not None else t_ms

        reason = ""
        if self.state != "FAULT" and self.t_bad_since is not None and t_ms - self.t_bad_since >= c.fault_ms:
            reason = self._go("FAULT", t_ms, "fsr out of range" if self.stall < c.logit_stall_ticks else "inference stalled")

        # 3. State machine
        s = self.state
        dt = t_ms - self.t_state
        if s == "FAULT":
            if self.t_good_since is not None and t_ms - self.t_good_since >= c.fault_clear_ms:
                reason = self._go("IDLE", t_ms, "fault cleared")

        elif s == "IDLE":
            if stance and stance_ms >= c.max_stance_ms:
                reason = self._go("STANDING", t_ms, "stance too long")
            elif (logit_valid and logit > c.logit_threshold
                  and stance and stance_ms >= c.min_stance_ms):
                self.t_logit_low = None
                reason = self._go("SPARK", t_ms, f"intent @ stance {stance_ms:.0f} ms")

        elif s == "STANDING":
            if toe_off or not stance:
                reason = self._go("IDLE", t_ms, "movement resumed")

        elif s == "SPARK":
            asking = logit_valid and logit > c.logit_threshold
            if asking:
                self.t_logit_low = None
            elif self.t_logit_low is None:
                self.t_logit_low = t_ms
            if heel_contact:
                reason = self._go("CANCEL", t_ms, "veto: heel contact")
            elif toe_off:
                reason = self._go("ASSIST", t_ms, "confirmed: toe-off")
            elif abs(gy_dps) >= c.gy_confirm_dps:
                reason = self._go("ASSIST", t_ms, f"confirmed: gyro {gy_dps:+.0f} dps")
            elif self.t_logit_low is not None and t_ms - self.t_logit_low >= c.logit_drop_ms:
                reason = self._go("CANCEL", t_ms, "model withdrew")
            elif dt >= c.max_spark_ms:
                reason = self._go("CANCEL", t_ms, "veto: spark held too long")

        elif s == "ASSIST":
            if heel_contact:
                reason = self._go("REFRACTORY", t_ms, "released: heel contact")
            elif dt >= c.max_assist_ms:
                reason = self._go("REFRACTORY", t_ms, "veto: assist timeout")

        elif s == "CANCEL":
            if dt >= c.cancel_refractory_ms:
                reason = self._go("IDLE", t_ms, "")

        elif s == "REFRACTORY":
            if dt >= c.refractory_ms:
                reason = self._go("IDLE", t_ms, "")

        torque = {"SPARK": c.spark_torque, "ASSIST": c.assist_torque}.get(self.state, 0.0)
        return Command(torque, self.state, reason, "STANCE" if stance else "SWING",
                       self.heel.loaded, self.toe.loaded)

    def _go(self, state, t_ms, reason):
        self.state, self.t_state = state, t_ms
        return f"{state}: {reason}" if reason else state
