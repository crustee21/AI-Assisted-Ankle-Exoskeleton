"""
can_servo.py -- CubeMars AK70-10 in SERVO mode over the Pmod CAN (MCP25625).

Drop-in replacement for AK70 (MIT mode) in can_mcp.py: same command()/poll_state()/
enable()/disable() interface, so run_afo.py only swaps the class.

Servo-mode protocol (CubeMars AK manual, VESC-derived), extended 29-bit IDs:
    TX id = (mode << 8) | motor_id
        mode 1  current loop     int32 BE = amps * 1000
        mode 2  current brake    int32 BE = amps * 1000
        mode 3  velocity loop    int32 BE = electrical RPM
        mode 5  set origin       1 byte: 0 temporary, 1 permanent
    RX id = 0x2900 | motor_id   (the motor broadcasts this on its own)
        int16 pos * 0.1 deg | int16 speed * 10 ERPM | int16 current * 0.01 A | int8 temp C | uint8 error

Notebook usage (same directory as can_mcp.py):
    %run can_servo.py
    spi = AxiSpi(ol.axi_quad_spi_0.mmio)
    can = Mcp25625(spi, rst_gpio=(ol.gpio_can.mmio, 0x00, 0)); can.init()
    motor = AKServo(can, motor_id=1, torque_ceiling=0.5)
    print(motor.poll_state())
"""
import math
import time

from can_mcp import AxiSpi, Mcp25625  # noqa: F401  (re-exported for %run convenience)

# MCP2515/25625 SPI instructions and registers used here
_TXB0CTRL = 0x30
_LOAD_TXB0_SIDH = 0x40
_RTS_TXB0 = 0x81
_RD_STATUS = 0xA0
_READ_RXB0 = 0x90
_READ_RXB1 = 0x94


def send_ext(can, eid, data, timeout_s=0.02):
    """Send one extended-ID (29-bit) data frame through TXB0."""
    t0 = time.perf_counter()
    while can.read(_TXB0CTRL)[0] & 0x08:                 # previous frame still pending
        if time.perf_counter() - t0 > timeout_s:
            can.bitmod(_TXB0CTRL, 0x08, 0x00)
            raise RuntimeError("CAN TX stuck (no ACK? termination / bit rate / motor off?)")
    hdr = [(eid >> 21) & 0xFF,
           ((eid >> 13) & 0xE0) | 0x08 | ((eid >> 16) & 0x03),   # EXIDE set
           (eid >> 8) & 0xFF,
           eid & 0xFF]
    can.spi.transfer([_LOAD_TXB0_SIDH, *hdr, len(data), *data])
    can.spi.transfer([_RTS_TXB0])


def recv_any(can):
    """Return (is_ext, id, data) for one pending frame, or None."""
    st = can.spi.transfer([_RD_STATUS, 0])[1]
    if st & 0x01:
        cmd = _READ_RXB0
    elif st & 0x02:
        cmd = _READ_RXB1
    else:
        return None
    f = can.spi.transfer([cmd] + [0] * 13)[1:]
    ext = bool(f[1] & 0x08)
    sid = (f[0] << 3) | (f[1] >> 5)
    eid = (sid << 18) | ((f[1] & 0x03) << 16) | (f[2] << 8) | f[3]
    dlc = f[4] & 0x0F
    return ext, (eid if ext else sid), list(f[5:5 + dlc])


def _i16(hi, lo):
    v = (hi << 8) | lo
    return v - 0x10000 if v & 0x8000 else v


class AKServo:
    GEAR = 10
    POLE_PAIRS = 21        # AK70-10 -- verify with the spin test (see notes)
    KT_OUT = 1.1           # N*m per A at the output, approx (8.3 N*m / 7.2 A rated) -- calibrate
    I_MAX = 20.0           # A, absolute clamp regardless of torque_ceiling

    def __init__(self, can, motor_id=1, torque_ceiling=0.5):
        self.can = can
        self.id = motor_id
        self.ceiling = float(torque_ceiling)     # hard software clamp on command(), N*m
        self.state = None   # (pos_deg, vel_rad_s_output, torque_nm_est, temp_c, err)

    # -- low level -----------------------------------------------------------
    def _send(self, mode, data):
        send_ext(self.can, (mode << 8) | self.id, data)

    @staticmethod
    def _i32(x):
        return list(int(round(x)).to_bytes(4, "big", signed=True))

    def current(self, amps):
        a = min(max(amps, -self.I_MAX), self.I_MAX)
        self._send(1, self._i32(a * 1000))
        return a

    def brake(self, amps):
        self._send(2, self._i32(abs(amps) * 1000))

    def velocity(self, rad_s_output):
        """Motor's own speed loop. NOT limited by torque_ceiling -- only by the motor's
        configured current limit. Keep speeds low, e-stop in hand."""
        erpm = rad_s_output * 60 / (2 * math.pi) * self.GEAR * self.POLE_PAIRS
        self._send(3, self._i32(erpm))

    def set_origin(self, permanent=False):
        self._send(5, [1 if permanent else 0])

    # -- AK70-compatible interface ------------------------------------------
    def enable(self):
        """Servo mode has no enter/exit frames; just make sure we start at zero current."""
        self.current(0.0)

    def disable(self):
        self.current(0.0)                        # zero current = motor coasts

    def zero(self):
        self.set_origin(permanent=False)

    def command(self, torque_nm, **_ignored):
        """Feed-forward output torque, clamped to the ceiling. Returns the torque sent."""
        t = min(max(torque_nm, -self.ceiling), self.ceiling)
        self.current(t / self.KT_OUT)
        return t

    def poll_state(self):
        """Drain the RX buffers; keep the latest status frame for this motor."""
        while (fr := recv_any(self.can)) is not None:
            ext, fid, d = fr
            if ext and fid == (0x2900 | self.id) and len(d) >= 8:
                pos_deg = _i16(d[0], d[1]) * 0.1
                erpm = _i16(d[2], d[3]) * 10
                amps = _i16(d[4], d[5]) * 0.01
                temp = d[6] - 256 if d[6] & 0x80 else d[6]
                vel = erpm / (self.POLE_PAIRS * self.GEAR) * 2 * math.pi / 60
                self.state = (pos_deg, vel, amps * self.KT_OUT, temp, d[7])
        return self.state


def run_for(motor, sec, torque_nm=None, vel=None, every=10):
    """Stream a torque (or velocity) command at ~100 Hz and print state; always ends at 0 A."""
    t_end, i = time.time() + sec, 0
    try:
        while time.time() < t_end:
            if vel is not None:
                motor.velocity(vel)
            else:
                motor.command(torque_nm or 0.0)
            s = motor.poll_state()
            if s and i % every == 0:
                print(f"pos {s[0]:+8.1f} deg  vel {s[1]:+6.2f} rad/s  "
                      f"tau~{s[2]:+5.2f} Nm  {s[3]:3d}C  err {s[4]}")
            i += 1
            time.sleep(0.01)
    finally:
        motor.current(0.0)
