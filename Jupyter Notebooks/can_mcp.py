"""
can_mcp.py -- Digilent Pmod CAN (MCP25625) over AXI Quad SPI, and the CubeMars
AK70-10 MIT-mode protocol on top of it.

Three layers, each usable on its own from a notebook:

    spi   = AxiSpi(ol.axi_quad_spi_0.mmio)        # raw SPI transfers
    can   = Mcp25625(spi, rst_gpio=...)           # CAN controller: init, send, recv
    motor = AK70(can, motor_id=1)                 # enter/exit MIT mode, send torque, read state

Bench-test order:  spi loopback -> can.init() returns True -> can.send() with a
second CAN node listening (or a scope on CANH/CANL) -> motor.enable() and read a
state frame back -> motor.command(torque=0.2) with NOTHING on the shaft.

Pmod CAN crystal is 20 MHz; bit timing below is 1 Mbit/s, sample point 70 %,
which is the AK70-10 default rate. SPI mode 0, <= 10 MHz.
"""
import time

# ---------------------------------------------------------------------------
# AXI Quad SPI (xilinx.com:ip:axi_quad_spi) register-level driver
# ---------------------------------------------------------------------------
class AxiSpi:
    SRR, SPICR, SPISR, SPIDTR, SPIDRR, SPISSR = 0x40, 0x60, 0x64, 0x68, 0x6C, 0x70
    # SPICR bits
    SPE, MASTER, CPOL, CPHA = 1 << 1, 1 << 2, 1 << 3, 1 << 4
    TXFIFO_RST, RXFIFO_RST, MANUAL_SS, INHIBIT = 1 << 5, 1 << 6, 1 << 7, 1 << 8
    # SPISR bits
    RX_EMPTY, TX_EMPTY = 1 << 0, 1 << 2

    def __init__(self, mmio, fifo_depth=16):
        self.m = mmio
        self.fifo_depth = fifo_depth
        self.m.write(self.SRR, 0x0A)                         # soft reset
        time.sleep(0.001)
        self._cr = self.SPE | self.MASTER | self.MANUAL_SS | self.INHIBIT   # mode 0
        self.m.write(self.SPICR, self._cr | self.TXFIFO_RST | self.RXFIFO_RST)
        self.m.write(self.SPISSR, 0xFFFFFFFF)                # all slaves deselected

    def transfer(self, tx):
        """Full-duplex transfer of up to fifo_depth bytes with CS held low; returns rx bytes."""
        assert len(tx) <= self.fifo_depth, "split into chunks <= FIFO depth"
        self.m.write(self.SPICR, self._cr | self.TXFIFO_RST | self.RXFIFO_RST)
        for b in tx:
            self.m.write(self.SPIDTR, b & 0xFF)
        self.m.write(self.SPISSR, 0xFFFFFFFE)                # select slave 0
        self.m.write(self.SPICR, self._cr & ~self.INHIBIT)   # go
        t0 = time.perf_counter()
        while not (self.m.read(self.SPISR) & self.TX_EMPTY):
            if time.perf_counter() - t0 > 0.01:
                raise RuntimeError("SPI transfer timeout")
        self.m.write(self.SPICR, self._cr)                   # inhibit
        self.m.write(self.SPISSR, 0xFFFFFFFF)                # deselect
        rx = []
        while not (self.m.read(self.SPISR) & self.RX_EMPTY):
            rx.append(self.m.read(self.SPIDRR) & 0xFF)
        return rx


# ---------------------------------------------------------------------------
# MCP2515/25625 CAN controller
# ---------------------------------------------------------------------------
class Mcp25625:
    # SPI instructions
    RESET, READ, WRITE, BITMOD, RD_STATUS = 0xC0, 0x03, 0x02, 0x05, 0xA0
    LOAD_TXB0, RTS_TXB0, READ_RXB0, READ_RXB1 = 0x40, 0x81, 0x90, 0x94
    # registers
    CANSTAT, CANCTRL, CNF3, CNF2, CNF1 = 0x0E, 0x0F, 0x28, 0x29, 0x2A
    CANINTE, CANINTF, EFLG, TXB0CTRL, RXB0CTRL, RXB1CTRL = 0x2B, 0x2C, 0x2D, 0x30, 0x60, 0x70
    MODE_CONFIG, MODE_NORMAL, MODE_LOOPBACK = 0x80, 0x00, 0x40
    # 20 MHz crystal, 1 Mbit/s: Tq = 100 ns, 10 Tq/bit = sync1 + prop2 + ps1 4 + ps2 3
    BITTIMING_1M = {CNF1: 0x00, CNF2: 0x99, CNF3: 0x02}

    def __init__(self, spi, rst_gpio=None):
        self.spi = spi
        self.rst = rst_gpio          # (mmio, offset, bit) driving the Pmod ~RST, or None

    # -- register access -----------------------------------------------------
    def reset(self):
        if self.rst:
            m, off, bit = self.rst
            m.write(off, 0); time.sleep(0.001); m.write(off, 1 << bit); time.sleep(0.005)
        self.spi.transfer([self.RESET]); time.sleep(0.005)

    def read(self, addr, n=1):
        return self.spi.transfer([self.READ, addr] + [0] * n)[2:]

    def write(self, addr, *vals):
        self.spi.transfer([self.WRITE, addr, *vals])

    def bitmod(self, addr, mask, val):
        self.spi.transfer([self.BITMOD, addr, mask, val])

    def set_mode(self, mode):
        self.bitmod(self.CANCTRL, 0xE0, mode)
        for _ in range(50):
            if (self.read(self.CANSTAT)[0] & 0xE0) == mode:
                return True
            time.sleep(0.001)
        return False

    # -- bring-up ---------------------------------------------------------------
    def init(self, loopback=False):
        """Reset, program 1 Mbit timing, accept all frames. Returns True on success."""
        self.reset()
        if not self.set_mode(self.MODE_CONFIG):
            return False
        for reg, v in self.BITTIMING_1M.items():
            self.write(reg, v)
        if self.read(self.CNF2)[0] != 0x99:
            return False                                    # SPI not talking to the chip
        self.write(self.RXB0CTRL, 0x60)                     # receive any, no filters
        self.write(self.RXB1CTRL, 0x60)
        self.write(self.CANINTE, 0x00)                      # we poll
        return self.set_mode(self.MODE_LOOPBACK if loopback else self.MODE_NORMAL)

    # -- frames -------------------------------------------------------------------
    def send(self, can_id, data, timeout_s=0.005):
        """Standard 11-bit frame. Blocks until TXB0 is free, then queues it."""
        t0 = time.perf_counter()
        while self.read(self.TXB0CTRL)[0] & 0x08:           # TXREQ still set
            if time.perf_counter() - t0 > timeout_s:
                self.bitmod(self.TXB0CTRL, 0x08, 0x00)      # abort the stuck one
                raise RuntimeError("CAN TX stuck (no ACK? bus not terminated / motor off?)")
        sidh, sidl = (can_id >> 3) & 0xFF, (can_id & 0x07) << 5
        self.spi.transfer([self.LOAD_TXB0, sidh, sidl, 0, 0, len(data), *data])
        self.spi.transfer([self.RTS_TXB0])

    def recv(self):
        """Return (can_id, data) of one pending frame, or None."""
        st = self.spi.transfer([self.RD_STATUS, 0])[1]
        if st & 0x01:
            f = self.spi.transfer([self.READ_RXB0] + [0] * 13)[1:]
        elif st & 0x02:
            f = self.spi.transfer([self.READ_RXB1] + [0] * 13)[1:]
        else:
            return None
        can_id = (f[0] << 3) | (f[1] >> 5)
        dlc = f[4] & 0x0F
        return can_id, f[5:5 + dlc]

    def errors(self):
        return self.read(self.EFLG)[0]


# ---------------------------------------------------------------------------
# CubeMars AK70-10, MIT mode
# ---------------------------------------------------------------------------
def _f2u(x, lo, hi, bits):
    x = min(max(x, lo), hi)
    return int((x - lo) * ((1 << bits) - 1) / (hi - lo))

def _u2f(u, lo, hi, bits):
    return u * (hi - lo) / ((1 << bits) - 1) + lo

class AK70:
    P_MIN, P_MAX = -12.5, 12.5        # rad
    V_MIN, V_MAX = -50.0, 50.0        # rad/s
    KP_MIN, KP_MAX = 0.0, 500.0
    KD_MIN, KD_MAX = 0.0, 5.0
    T_MIN, T_MAX = -25.0, 25.0        # N*m  (AK70-10 MIT-mode range)

    def __init__(self, can, motor_id=1, torque_ceiling=2.0):
        self.can = can
        self.id = motor_id
        self.ceiling = float(torque_ceiling)   # hard software clamp, N*m -- raise deliberately
        self.state = None                      # (pos_rad, vel_rads, torque_nm) from last reply

    def _magic(self, last):
        self.can.send(self.id, [0xFF] * 7 + [last])

    def enable(self):  self._magic(0xFC)       # enter MIT motor mode
    def disable(self): self._magic(0xFD)       # exit (motor goes limp)
    def zero(self):    self._magic(0xFE)       # set current position as zero

    def command(self, torque_nm, pos=0.0, vel=0.0, kp=0.0, kd=0.0):
        """Pure feed-forward torque by default (kp = kd = 0). Clamped to the ceiling."""
        t = min(max(torque_nm, -self.ceiling), self.ceiling)
        p, v = _f2u(pos, self.P_MIN, self.P_MAX, 16), _f2u(vel, self.V_MIN, self.V_MAX, 12)
        kpu, kdu = _f2u(kp, self.KP_MIN, self.KP_MAX, 12), _f2u(kd, self.KD_MIN, self.KD_MAX, 12)
        tu = _f2u(t, self.T_MIN, self.T_MAX, 12)
        data = [p >> 8, p & 0xFF, v >> 4, ((v & 0xF) << 4) | (kpu >> 8), kpu & 0xFF,
                kdu >> 4, ((kdu & 0xF) << 4) | (tu >> 8), tu & 0xFF]
        self.can.send(self.id, data)
        return t

    def poll_state(self):
        """Drain replies; keep the latest (pos, vel, torque). Returns it or None."""
        while True:
            f = self.can.recv()
            if f is None:
                return self.state
            _, d = f
            if len(d) >= 6 and d[0] == self.id:
                pu = (d[1] << 8) | d[2]
                vu = (d[3] << 4) | (d[4] >> 4)
                tu = ((d[4] & 0xF) << 8) | d[5]
                self.state = (_u2f(pu, self.P_MIN, self.P_MAX, 16),
                              _u2f(vu, self.V_MIN, self.V_MAX, 12),
                              _u2f(tu, self.T_MIN, self.T_MAX, 12))


class NoMotor:
    """Drop-in stand-in so run_afo.py runs identically with nothing on the bus."""
    def __init__(self, torque_ceiling=2.0): self.ceiling = torque_ceiling; self.state = None
    def enable(self): pass
    def disable(self): pass
    def zero(self): pass
    def command(self, torque_nm, **kw): return min(max(torque_nm, -self.ceiling), self.ceiling)
    def poll_state(self): return None
