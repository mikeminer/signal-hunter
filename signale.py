import sys
import json
import asyncio
import threading
import time
import winsound
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd
import websockets

from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QPushButton, QSpinBox, QComboBox,
    QGroupBox, QFormLayout
)

# =========================
# DATA STRUCTURES
# =========================

@dataclass
class SignalOut:
    ts: int
    side: str
    entry: float
    stop: float
    add_zone: Tuple[float, float]
    reason: str
    prob: int  # 0..100

@dataclass
class Zones:
    # list of (level, touches)
    eq_highs: List[Tuple[float, int]]
    eq_lows: List[Tuple[float, int]]

# =========================
# CORE LOGIC
# =========================

def anchored_vwap(df: pd.DataFrame) -> float:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_v = df["volume"].cumsum()
    cum_vtp = (tp * df["volume"]).cumsum()
    vwap = (cum_vtp / cum_v.replace(0, np.nan)).iloc[-1]
    return float(vwap) if np.isfinite(vwap) else float("nan")

def find_equal_levels_with_touches(levels: np.ndarray, tol_pct: float = 0.0010, min_touches: int = 3) -> List[Tuple[float, int]]:
    """
    Cluster swing highs/lows into equal levels with touch counts.
    Returns list of (cluster_center, touches)
    """
    levels = np.asarray(levels, dtype=float)
    if len(levels) < min_touches:
        return []

    levels_sorted = np.sort(levels)
    clusters: List[Tuple[float, int]] = []
    current = [levels_sorted[0]]

    for x in levels_sorted[1:]:
        ref = float(np.mean(current))
        if abs(x - ref) / ref <= tol_pct:
            current.append(x)
        else:
            if len(current) >= min_touches:
                clusters.append((float(np.mean(current)), len(current)))
            current = [x]

    if len(current) >= min_touches:
        clusters.append((float(np.mean(current)), len(current)))

    # De-dup close cluster centers, keeping max touches
    out: List[Tuple[float, int]] = []
    for c, t in sorted(clusters, key=lambda z: z[0]):
        if not out:
            out.append((c, t))
        else:
            prev_c, prev_t = out[-1]
            if abs(c - prev_c) / prev_c <= tol_pct:
                out[-1] = (float((prev_c + c) / 2.0), max(prev_t, t))
            else:
                out.append((c, t))
    return out

def detect_zones(df: pd.DataFrame, swing_lookback: int = 3) -> Zones:
    highs = df["high"].values
    lows = df["low"].values

    swing_highs = []
    swing_lows = []
    for i in range(swing_lookback, len(df) - swing_lookback):
        if highs[i] == np.max(highs[i - swing_lookback:i + swing_lookback + 1]):
            swing_highs.append(highs[i])
        if lows[i] == np.min(lows[i - swing_lookback:i + swing_lookback + 1]):
            swing_lows.append(lows[i])

    eq_highs = find_equal_levels_with_touches(np.array(swing_highs), tol_pct=0.0010, min_touches=3)
    eq_lows  = find_equal_levels_with_touches(np.array(swing_lows),  tol_pct=0.0010, min_touches=3)
    return Zones(eq_highs=eq_highs, eq_lows=eq_lows)

def clamp_int(x: float, lo: int = 0, hi: int = 100) -> int:
    return int(max(lo, min(hi, round(x))))

def compute_probability_score(
    touches: int,
    funding: Optional[float],
    rejection_ratio: float,
    swept_pct: float,
    conservative_reclaim: bool
) -> int:
    """
    Euristica 0..100 (non è “verità”), utile per filtrare:
    - touches: quante volte il livello è stato toccato (più = più stop)
    - funding magnitude: crowding bias
    - rejection_ratio: wick/body (più alto = rifiuto più forte)
    - swept_pct: profondità dello sweep in % (più = più “grab”)
    - conservative_reclaim: reclaim confermato = più affidabile
    """
    score = 35.0

    # Touches (fino a +30)
    score += min(30.0, max(0.0, (touches - 2) * 10.0))  # 3 touches => +10, 4 => +20, 5+ => +30

    # Funding magnitude (fino a +15)
    if funding is not None:
        score += min(15.0, abs(funding) * 100000.0)  # es. 0.00020 -> +20 (cappato 15)

    # Rejection ratio (fino a +15)
    score += min(15.0, max(0.0, rejection_ratio) * 6.0)  # ratio ~2.5 => +15

    # Sweep depth (fino a +10)
    score += min(10.0, max(0.0, swept_pct * 1000.0))  # 0.2% => +2

    # Reclaim confermato (fino a +10)
    if conservative_reclaim:
        score += 10.0

    return clamp_int(score, 0, 100)

def detect_liquidity_grab_signal(
    df: pd.DataFrame,
    zones: Zones,
    funding: Optional[float],
    aggressive: bool
) -> Optional[SignalOut]:
    if len(df) < 220:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    hi = float(last["high"])
    lo = float(last["low"])
    op = float(last["open"])
    ts = int(last["ts"])

    vwap = anchored_vwap(df)
    if not np.isfinite(vwap):
        return None

    # Candle anatomy
    body = abs(price - op)
    body = body if body > 1e-9 else 1e-9
    upper_wick = hi - max(price, op)
    lower_wick = min(price, op) - lo

    def strong_rejection_short() -> bool:
        return upper_wick > body * 1.2

    def strong_rejection_long() -> bool:
        return lower_wick > body * 1.2

    # Funding bias filter (se non disponibile, lasciamo passare)
    def bias_ok(side: str) -> bool:
        if funding is None:
            return True
        return (side == "SHORT" and funding >= 0.0) or (side == "LONG" and funding <= 0.0)

    # Pick nearest relevant levels
    def nearest(levels: List[Tuple[float, int]], ref: float) -> Optional[Tuple[float, int]]:
        if not levels:
            return None
        return sorted(levels, key=lambda x: abs(x[0] - ref))[0]

    near_eq_high = nearest(zones.eq_highs, price)
    near_eq_low = nearest(zones.eq_lows, price)

    # SHORT: sweep above eqHigh + reclaim below (or aggressive: strong rejection)
    if near_eq_high is not None:
        lvl, touches = near_eq_high
        swept = hi > lvl * 1.0003
        reclaim = price < lvl
        aggressive_ok = aggressive and strong_rejection_short()

        if swept and (reclaim or aggressive_ok) and bias_ok("SHORT"):
            entry = price
            stop = hi * 1.0008
            add1 = float(min(vwap, lvl))
            add2 = float(max(vwap, lvl))

            rejection_ratio = upper_wick / body
            swept_pct = (hi - lvl) / lvl  # how far above lvl we swept

            prob = compute_probability_score(
                touches=touches,
                funding=funding,
                rejection_ratio=rejection_ratio,
                swept_pct=swept_pct,
                conservative_reclaim=reclaim
            )

            return SignalOut(
                ts=ts, side="SHORT", entry=entry, stop=stop, add_zone=(add1, add2),
                reason=f"Sweep+Reject above eqHigh {lvl:.2f} (touches={touches}) | funding={funding}",
                prob=prob
            )

    # LONG: sweep below eqLow + reclaim above (or aggressive: strong rejection)
    if near_eq_low is not None:
        lvl, touches = near_eq_low
        swept = lo < lvl * 0.9997
        reclaim = price > lvl
        aggressive_ok = aggressive and strong_rejection_long()

        if swept and (reclaim or aggressive_ok) and bias_ok("LONG"):
            entry = price
            stop = lo * 0.9992
            add1 = float(min(vwap, lvl))
            add2 = float(max(vwap, lvl))

            rejection_ratio = lower_wick / body
            swept_pct = (lvl - lo) / lvl  # how far below lvl we swept

            prob = compute_probability_score(
                touches=touches,
                funding=funding,
                rejection_ratio=rejection_ratio,
                swept_pct=swept_pct,
                conservative_reclaim=reclaim
            )

            return SignalOut(
                ts=ts, side="LONG", entry=entry, stop=stop, add_zone=(add1, add2),
                reason=f"Sweep+Reject below eqLow {lvl:.2f} (touches={touches}) | funding={funding}",
                prob=prob
            )

    return None

# =========================
# REALTIME WS WORKER
# =========================

class RealtimeWorker(QObject):
    price_update = pyqtSignal(float)
    funding_update = pyqtSignal(object)     # float or None
    zones_update = pyqtSignal(object)       # Zones
    signal_found = pyqtSignal(object)       # SignalOut
    status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._stop = False
        self.max_candles = 800
        self.aggressive = False

        self.candles = deque(maxlen=self.max_candles)
        self.funding: Optional[float] = None
        self.last_signal_ts: Optional[int] = None

    def configure(self, max_candles: int, aggressive: bool):
        self.max_candles = max_candles
        self.candles = deque(self.candles, maxlen=max_candles)
        self.aggressive = aggressive

    def stop(self):
        self._stop = True

    async def run(self):
        url = "wss://fstream.binance.com/stream?streams=ethusdt@kline_1m/ethusdt@markPrice@1s"
        self.status.emit("Connecting...")

        while not self._stop:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    self.status.emit("Connected. Streaming realtime...")
                    async for raw in ws:
                        if self._stop:
                            break

                        msg = json.loads(raw)
                        stream = msg.get("stream", "")
                        data = msg.get("data", {})

                        if "markPrice" in stream:
                            try:
                                self.price_update.emit(float(data.get("p")))
                            except Exception:
                                pass
                            try:
                                r = data.get("r")
                                self.funding = float(r) if r is not None else None
                                self.funding_update.emit(self.funding)
                            except Exception:
                                self.funding = None
                                self.funding_update.emit(None)

                        elif "kline" in stream:
                            k = data.get("k", {})
                            is_closed = bool(k.get("x", False))

                            if is_closed:
                                ts = int(k.get("t", 0))
                                o = float(k.get("o", 0))
                                h = float(k.get("h", 0))
                                l = float(k.get("l", 0))
                                c = float(k.get("c", 0))
                                v = float(k.get("v", 0))

                                self.candles.append((ts, o, h, l, c, v))
                                self.price_update.emit(c)

                                if len(self.candles) >= 220:
                                    df = pd.DataFrame(list(self.candles),
                                                      columns=["ts", "open", "high", "low", "close", "volume"])
                                    zones = detect_zones(df)
                                    self.zones_update.emit(zones)

                                    sig = detect_liquidity_grab_signal(df, zones, self.funding, aggressive=self.aggressive)
                                    if sig and sig.ts != self.last_signal_ts:
                                        self.last_signal_ts = sig.ts
                                        self.signal_found.emit(sig)

            except Exception as e:
                self.status.emit(f"WS error: {repr(e)} | reconnecting...")
                await asyncio.sleep(2)

        self.status.emit("Stopped streaming.")

def start_async_worker(worker: RealtimeWorker):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(worker.run())
    loop.close()

# =========================
# SIREN (continuous) CONTROL
# =========================

class SirenController:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active = False

    def start(self):
        if self._active:
            return
        self._stop_event.clear()
        self._active = True

        def run():
            # Pattern “forte” finché non viene stoppata
            try:
                winsound.MessageBeep(winsound.MB_ICONHAND)
            except Exception:
                pass

            while not self._stop_event.is_set():
                try:
                    # “Sirena”: alterna frequenze
                    winsound.Beep(1400, 250)
                    if self._stop_event.is_set():
                        break
                    winsound.Beep(900, 250)
                    time.sleep(0.05)
                except Exception:
                    # Se winsound fallisse, evita crash
                    time.sleep(0.2)

            self._active = False

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

# =========================
# UI
# =========================

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ETH Realtime Liquidity Grab Signals (WebSocket)")
        self.resize(920, 720)

        self.worker = RealtimeWorker()
        self.worker_thread: Optional[threading.Thread] = None

        # Siren + alert control
        self.siren = SirenController()
        self.last_alert_time = 0.0
        self.muted_by_ack = False  # ACK silenzia fino al prossimo segnale (ma senza sirena)
        # (puoi cambiarlo per “muted finché non premi un toggle”, se vuoi)

        layout = QVBoxLayout()

        controls = QHBoxLayout()
        self.maxcand = QSpinBox()
        self.maxcand.setRange(250, 2000)
        self.maxcand.setValue(800)

        self.mode = QComboBox()
        self.mode.addItems(["Conservative (after reclaim)", "Aggressive (during sweep)"])

        self.cooldown = QSpinBox()
        self.cooldown.setRange(0, 3600)
        self.cooldown.setValue(60)  # default: 60s anti-spam

        self.start_btn = QPushButton("Start realtime")
        self.stop_btn = QPushButton("Stop")
        self.ack_btn = QPushButton("ACK (silenzia)")
        self.stop_btn.setEnabled(False)
        self.ack_btn.setEnabled(True)

        controls.addWidget(QLabel("Rolling candles:"))
        controls.addWidget(self.maxcand)
        controls.addWidget(QLabel("Mode:"))
        controls.addWidget(self.mode)
        controls.addWidget(QLabel("Cooldown (s):"))
        controls.addWidget(self.cooldown)
        controls.addStretch()
        controls.addWidget(self.ack_btn)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        layout.addLayout(controls)

        status_box = QGroupBox("Live")
        form = QFormLayout()
        self.price_lbl = QLabel("-")
        self.funding_lbl = QLabel("-")
        self.eqh_lbl = QLabel("-")
        self.eql_lbl = QLabel("-")
        self.state_lbl = QLabel("-")
        self.alarm_lbl = QLabel("OFF")

        form.addRow("Price:", self.price_lbl)
        form.addRow("Funding:", self.funding_lbl)
        form.addRow("Equal highs:", self.eqh_lbl)
        form.addRow("Equal lows:", self.eql_lbl)
        form.addRow("Alarm:", self.alarm_lbl)
        form.addRow("Status:", self.state_lbl)

        status_box.setLayout(form)
        layout.addWidget(status_box)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, stretch=1)

        self.setLayout(layout)

        # worker signals
        self.worker.price_update.connect(self.on_price)
        self.worker.funding_update.connect(self.on_funding)
        self.worker.zones_update.connect(self.on_zones)
        self.worker.signal_found.connect(self.on_signal)
        self.worker.status.connect(self.on_status)

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.ack_btn.clicked.connect(self.ack)

    def logline(self, s: str):
        self.log.append(s)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def on_price(self, p: float):
        self.price_lbl.setText(f"{p:.2f}")

    def on_funding(self, f):
        self.funding_lbl.setText("N/A" if f is None else f"{float(f):.6f}")

    def on_zones(self, z: Zones):
        if z.eq_highs:
            txt = ", ".join([f"{lvl:.2f}({t})" for (lvl, t) in z.eq_highs[-5:]])
        else:
            txt = "-"
        self.eqh_lbl.setText(txt)

        if z.eq_lows:
            txt = ", ".join([f"{lvl:.2f}({t})" for (lvl, t) in z.eq_lows[-5:]])
        else:
            txt = "-"
        self.eql_lbl.setText(txt)

    def on_status(self, s: str):
        self.state_lbl.setText(s)

    def ack(self):
        # Silenzia subito sirena, ma NON ferma lo streaming
        self.siren.stop()
        self.alarm_lbl.setText("OFF (ACK)")
        self.muted_by_ack = True
        self.logline("ACK: allarme silenziato. (Riparte al prossimo segnale se passa il cooldown)")

    def start(self):
        aggressive = (self.mode.currentIndex() == 1)
        self.worker.configure(max_candles=int(self.maxcand.value()), aggressive=aggressive)
        self.worker._stop = False

        self.worker_thread = threading.Thread(target=start_async_worker, args=(self.worker,), daemon=True)
        self.worker_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.logline("Started realtime stream.")

    def stop(self):
        # Stop streaming + stop siren
        self.worker.stop()
        self.siren.stop()
        self.alarm_lbl.setText("OFF")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.logline("Stopping...")

    def on_signal(self, s: SignalOut):
        now = time.time()
        cd = int(self.cooldown.value())

        # Log sempre il segnale
        self.logline(
            f"\n🚨 SIGNAL {s.side} | PROB={s.prob}%\n"
            f"entry={s.entry:.2f} stop={s.stop:.2f} add_zone={s.add_zone[0]:.2f}-{s.add_zone[1]:.2f}\n"
            f"Reason: {s.reason}\n"
        )

        # Cooldown anti-spam: se troppo presto, non suonare
        if cd > 0 and (now - self.last_alert_time) < cd:
            self.logline(f"Cooldown attivo ({cd}s): allarme NON riprodotto.")
            return

        # Se era stato premuto ACK, permetti comunque al prossimo segnale di riattivare,
        # ma solo se non siamo in cooldown
        self.muted_by_ack = False

        # Start siren continua
        self.last_alert_time = now
        self.siren.start()
        self.alarm_lbl.setText("ON 🔊 (SIREN)")
        self.logline("ALLARME: sirena attiva (premi ACK o STOP).")


# =========================
# MAIN
# =========================

def main():
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
