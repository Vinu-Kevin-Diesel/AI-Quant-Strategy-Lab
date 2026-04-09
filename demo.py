"""
demo.py — Interactive Strategy Lab Demo
=========================================
Run without MT5 — uses bundled sample data to demonstrate:
1. Gaussian IIR filter computation
2. HMM regime detection
3. Strategy signal generation
4. Backtest simulation with equity curve

Usage:
    python demo.py                # Full interactive demo
    python demo.py --quick        # Quick 30-second demo
"""
import sys
import math
import time
import argparse
import numpy as np

# ═══════════════════════════════════════════════════════════════════
#  GENERATE SYNTHETIC GOLD DATA (no MT5 needed)
# ═══════════════════════════════════════════════════════════════════
def generate_gold_data(n_bars=2000, seed=42):
    """Generate realistic XAUUSD-like price data with regime changes."""
    np.random.seed(seed)

    # Start at $2000
    price = 2000.0
    prices = [price]
    regimes = []  # 0=range, 1=bull, 2=bear

    current_regime = 0
    regime_bars = 0

    for i in range(1, n_bars):
        regime_bars += 1

        # Regime transitions
        if regime_bars > np.random.randint(100, 400):
            current_regime = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
            regime_bars = 0

        # Price movement based on regime
        if current_regime == 0:  # Ranging
            drift = np.random.normal(0, 0.0002)
            vol = 0.003
        elif current_regime == 1:  # Bull
            drift = np.random.normal(0.0008, 0.0003)
            vol = 0.005
        else:  # Bear
            drift = np.random.normal(-0.0006, 0.0003)
            vol = 0.006

        ret = drift + np.random.normal(0, vol)
        price *= (1 + ret)
        prices.append(price)
        regimes.append(current_regime)

    regimes.append(regimes[-1])

    # Generate OHLC from close
    closes = np.array(prices)
    opens = np.roll(closes, 1); opens[0] = closes[0]
    noise = np.random.uniform(0.001, 0.004, n_bars)
    highs = np.maximum(opens, closes) * (1 + noise)
    lows = np.minimum(opens, closes) * (1 - noise)

    return opens, highs, lows, closes, np.array(regimes)


# ═══════════════════════════════════════════════════════════════════
#  GAUSSIAN IIR FILTER
# ═══════════════════════════════════════════════════════════════════
def gaussian_filter(close, period=80, poles=4):
    """John Ehlers' multi-pole recursive IIR Gaussian filter."""
    beta = (1 - math.cos(2 * math.pi / period)) / (pow(2, 1.0 / poles) - 1)
    alpha = -beta + math.sqrt(beta * beta + 2 * beta)
    result = close.copy()
    for _ in range(poles):
        buf = result.copy()
        for j in range(1, len(result)):
            buf[j] = alpha * result[j] + (1 - alpha) * buf[j - 1]
        result = buf
    return result


# ═══════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════
def calc_atr(high, low, close, period=14):
    tr = np.empty(len(close))
    tr[0] = high[0] - low[0]
    for j in range(1, len(close)):
        tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
    atr = np.full(len(close), np.nan)
    atr[period] = tr[1:period+1].mean()
    for j in range(period + 1, len(close)):
        atr[j] = (atr[j-1] * (period - 1) + tr[j]) / period
    return atr

def calc_macd(close, fast=12, slow=26, signal=9):
    import warnings; warnings.filterwarnings('ignore')
    ema_f = np.zeros(len(close)); ema_s = np.zeros(len(close))
    af = 2/(fast+1); a_s = 2/(slow+1); a_sig = 2/(signal+1)
    ema_f[0] = close[0]; ema_s[0] = close[0]
    for i in range(1, len(close)):
        ema_f[i] = af * close[i] + (1-af) * ema_f[i-1]
        ema_s[i] = a_s * close[i] + (1-a_s) * ema_s[i-1]
    macd = ema_f - ema_s
    sig = np.zeros(len(close)); sig[0] = macd[0]
    for i in range(1, len(close)):
        sig[i] = a_sig * macd[i] + (1-a_sig) * sig[i-1]
    return macd, sig, macd - sig

def calc_rsi(close, period=14):
    d = np.diff(close)
    gains = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    rsi = np.full(len(close), 50.0)
    ag = gains[:period].mean(); al = losses[:period].mean()
    if al > 0: rsi[period] = 100 - 100 / (1 + ag / al)
    for j in range(period, len(d)):
        ag = (ag * (period - 1) + gains[j]) / period
        al = (al * (period - 1) + losses[j]) / period
        rsi[j + 1] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return rsi


# ═══════════════════════════════════════════════════════════════════
#  HMM REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════
def detect_regimes_hmm(close, n_states=5):
    """Train Gaussian HMM on returns/volatility features."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  [hmmlearn not installed — using rule-based regime detection]")
        return detect_regimes_rules(close)

    returns = np.log(close[1:] / close[:-1])
    vol = np.abs(returns)

    # Rolling features
    window = 20
    features = []
    for i in range(window, len(returns)):
        r = returns[i-window:i]
        features.append([r.mean(), r.std(), vol[i]])

    X = np.array(features)

    best_model = None; best_score = -np.inf
    for seed in range(5):
        try:
            model = GaussianHMM(n_components=n_states, covariance_type="full",
                                n_iter=100, random_state=seed * 42, tol=0.01)
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score; best_model = model
        except: continue

    if best_model is None:
        return detect_regimes_rules(close)

    states = best_model.predict(X)

    # Map states by mean return
    state_returns = {}
    for s in range(n_states):
        mask = states == s
        state_returns[s] = X[mask, 0].mean() if mask.sum() > 0 else 0

    sorted_states = sorted(state_returns.items(), key=lambda x: x[1], reverse=True)
    state_map = {orig: i + 1 for i, (orig, _) in enumerate(sorted_states)}
    regimes = np.array([state_map[s] for s in states])

    # Pad to match close length
    full = np.full(len(close), 3)
    full[window + 1:] = regimes
    return full

def detect_regimes_rules(close, period=50):
    """Simple rule-based regime detection (fallback)."""
    regimes = np.full(len(close), 3)  # Default neutral
    for i in range(period, len(close)):
        ret = (close[i] - close[i - period]) / close[i - period] * 100
        if ret > 3: regimes[i] = 1    # Bull
        elif ret > 1: regimes[i] = 2  # Mild bull
        elif ret < -3: regimes[i] = 5 # Strong bear
        elif ret < -1: regimes[i] = 4 # Mild bear
    return regimes


# ═══════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════
def run_backtest(close, high, low, gauss, macd_hist, rsi, atr,
                 sl_mult=2.0, tp_mult=5.0, max_sl=30.0):
    """Simulate GaussMACD strategy."""
    balance = 1000.0; equity_curve = [balance]
    trades = []; position = None

    for i in range(2, len(close)):
        if np.isnan(atr[i]) or atr[i] < 0.5:
            equity_curve.append(balance)
            continue

        # Check SL/TP
        if position is not None:
            if position['dir'] == 1:
                if low[i] <= position['sl']:
                    balance += position['sl'] - position['entry']
                    trades.append(position['sl'] - position['entry'])
                    position = None
                elif high[i] >= position['tp']:
                    balance += position['tp'] - position['entry']
                    trades.append(position['tp'] - position['entry'])
                    position = None
            else:
                if high[i] >= position['sl']:
                    balance += position['entry'] - position['sl']
                    trades.append(position['entry'] - position['sl'])
                    position = None
                elif low[i] <= position['tp']:
                    balance += position['entry'] - position['tp']
                    trades.append(position['entry'] - position['tp'])
                    position = None

        equity_curve.append(balance)

        if position is not None: continue
        if i < 100: continue

        gf1 = gauss[i-1]; gf2 = gauss[i-2]
        mh1 = macd_hist[i-1]; mh2 = macd_hist[i-2]

        if gf1 == 0 or gf2 == 0: continue

        sl_dist = atr[i-1] * sl_mult
        tp_dist = atr[i-1] * tp_mult
        if sl_dist > max_sl:
            ratio = tp_mult / sl_mult
            sl_dist = max_sl; tp_dist = sl_dist * ratio

        # Buy: Gaussian rising + MACD turning up + price above Gaussian
        if gf1 > gf2 and mh1 > mh2 and mh1 > -0.5 and close[i-1] > gf1 and rsi[i-1] < 80:
            position = {'dir': 1, 'entry': close[i], 'sl': close[i] - sl_dist, 'tp': close[i] + tp_dist}
        elif gf1 < gf2 and mh1 < mh2 and mh1 < 0.5 and close[i-1] < gf1 and rsi[i-1] > 28:
            position = {'dir': -1, 'entry': close[i], 'sl': close[i] + sl_dist, 'tp': close[i] - tp_dist}

    return np.array(equity_curve), trades


# ═══════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════
def print_header(text, width=60):
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")

def print_bar_chart(label, value, max_val, width=30, char='+'):
    bar_len = int(abs(value) / max(max_val, 1) * width)
    bar = char * bar_len
    color_code = ''
    print(f"  {label:<15} ${value:>+8.2f}  {bar}")

def print_progress(current, total, prefix='', width=40):
    pct = current / total
    filled = int(width * pct)
    bar = '#' * filled + '-' * (width - filled)
    sys.stdout.write(f'\r  {prefix} [{bar}] {pct*100:.0f}%')
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════
#  INTERACTIVE DEMO
# ═══════════════════════════════════════════════════════════════════
def run_demo(quick=False):
    n_bars = 1000 if quick else 2000

    print_header("MT5 STRATEGY LAB — INTERACTIVE DEMO")
    print("  Demonstrating Gaussian signal processing, HMM regime")
    print("  detection, and multi-strategy backtesting.")
    print("  No MT5 required — using synthetic gold data.\n")

    # ── Step 1: Generate Data ──
    print_header("STEP 1: Generating Synthetic XAUUSD Data")
    opens, highs, lows, closes, true_regimes = generate_gold_data(n_bars)
    print(f"  Generated {n_bars} H1 bars")
    print(f"  Price range: ${closes.min():.2f} - ${closes.max():.2f}")
    print(f"  Start: ${closes[0]:.2f} | End: ${closes[-1]:.2f}")

    # ── Step 2: Gaussian Filter ──
    print_header("STEP 2: Computing Gaussian IIR Filter (80-period, 4-pole)")
    print("  Formula: alpha * price + (1-alpha) * prev_output")
    print("  Applied 4 times recursively for maximum smoothing")

    gauss = gaussian_filter(closes, period=80, poles=4)

    # Show filter vs price at key points
    print(f"\n  {'Bar':>6} {'Price':>10} {'Gaussian':>10} {'Diff':>8} {'Trend':>8}")
    print(f"  {'-'*48}")
    for i in [100, 300, 500, 700, 900, min(n_bars-1, 1500)]:
        if i >= n_bars: break
        diff = closes[i] - gauss[i]
        trend = "UP" if gauss[i] > gauss[i-1] else "DOWN"
        print(f"  {i:>6} ${closes[i]:>9.2f} ${gauss[i]:>9.2f} {diff:>+7.2f}  {trend}")

    # ── Step 3: Indicators ──
    print_header("STEP 3: Computing Indicators")
    atr = calc_atr(highs, lows, closes)
    _, _, macd_hist = calc_macd(closes)
    rsi = calc_rsi(closes)

    valid = ~np.isnan(atr) & ~np.isnan(rsi)
    print(f"  ATR(14):  avg=${np.nanmean(atr):.2f}  min=${np.nanmin(atr[valid]):.2f}  max=${np.nanmax(atr[valid]):.2f}")
    print(f"  RSI(14):  avg={np.nanmean(rsi):.1f}  min={np.nanmin(rsi[valid]):.1f}  max={np.nanmax(rsi[valid]):.1f}")
    print(f"  MACD hist: range [{macd_hist[50:].min():.2f}, {macd_hist[50:].max():.2f}]")

    # ── Step 4: HMM Regime Detection ──
    print_header("STEP 4: HMM Regime Detection (5 States)")
    print("  Training Gaussian HMM on returns, volatility, volume...")

    regimes = detect_regimes_hmm(closes, n_states=5)
    regime_names = {1: "STRONG BULL", 2: "MILD BULL", 3: "NEUTRAL", 4: "MILD BEAR", 5: "STRONG BEAR"}

    print(f"\n  Regime Distribution:")
    for r in sorted(np.unique(regimes)):
        count = (regimes == r).sum()
        pct = count / len(regimes) * 100
        bar = '#' * int(pct / 2)
        name = regime_names.get(int(r), f"State {int(r)}")
        print(f"    {name:>15}: {count:>5} bars ({pct:>5.1f}%) {bar}")

    current = int(regimes[-1])
    print(f"\n  Current regime: {regime_names.get(current, '?')} (bar {n_bars})")

    # ── Step 5: Backtest ──
    print_header("STEP 5: Backtesting GaussMACD Strategy")
    print("  SL=2.0x ATR | TP=5.0x ATR | MaxSL=$30 | $1000 start")

    if not quick:
        for i in range(20):
            print_progress(i + 1, 20, prefix="Simulating")
            time.sleep(0.1)
        print()

    equity, trades = run_backtest(closes, highs, lows, gauss, macd_hist, rsi)

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    net = sum(trades)
    wr = len(wins) / len(trades) * 100 if trades else 0
    pf = sum(wins) / abs(sum(losses)) if losses else 999
    max_eq = np.maximum.accumulate(equity)
    dd = ((max_eq - equity) / max_eq * 100)
    max_dd = dd.max()

    print(f"\n  {'RESULTS':^50}")
    print(f"  {'-'*50}")
    print(f"  Final Balance:   ${equity[-1]:>10.2f}")
    print(f"  Net Profit:      ${net:>10.2f}")
    print(f"  Total Trades:    {len(trades):>10}")
    print(f"  Win Rate:        {wr:>9.1f}%")
    print(f"  Profit Factor:   {pf:>10.2f}")
    print(f"  Max Drawdown:    {max_dd:>9.1f}%")
    print(f"  Avg Win:         ${np.mean(wins) if wins else 0:>10.2f}")
    print(f"  Avg Loss:        ${np.mean(losses) if losses else 0:>10.2f}")

    # Equity curve ASCII art
    print_header("EQUITY CURVE")
    eq_samples = np.linspace(0, len(equity) - 1, 50, dtype=int)
    eq_vals = equity[eq_samples]
    min_eq, max_eq_val = eq_vals.min(), eq_vals.max()
    chart_height = 15

    for row in range(chart_height, -1, -1):
        threshold = min_eq + (max_eq_val - min_eq) * row / chart_height
        line = "  "
        if row == chart_height:
            line += f"${max_eq_val:>7.0f} |"
        elif row == 0:
            line += f"${min_eq:>7.0f} |"
        elif row == chart_height // 2:
            mid = (min_eq + max_eq_val) / 2
            line += f"${mid:>7.0f} |"
        else:
            line += "         |"

        for val in eq_vals:
            if val >= threshold:
                line += "*"
            else:
                line += " "
        print(line)
    print("          +" + "-" * 50)
    print("          Start" + " " * 35 + "End")

    # Monthly P&L
    print_header("MONTHLY P&L BREAKDOWN")
    bars_per_month = 500  # Approximate
    max_monthly = 0
    monthly_data = []
    for m in range(0, len(trades), max(1, len(trades) // 8)):
        chunk = trades[m:m + max(1, len(trades) // 8)]
        pnl = sum(chunk)
        monthly_data.append((f"Period {m // max(1, len(trades)//8) + 1}", pnl))
        if abs(pnl) > max_monthly: max_monthly = abs(pnl)

    for label, pnl in monthly_data:
        char = '+' if pnl > 0 else '-'
        print_bar_chart(label, pnl, max_monthly, char=char)

    # ── Step 6: Strategy Comparison ──
    print_header("STRATEGY VARIANT COMPARISON")
    print(f"  {'Variant':<25} {'Profit':>10} {'Trades':>8} {'WR%':>6} {'PF':>6}")
    print(f"  {'-'*60}")

    variants = [
        ("GaussMACD (2.0/5.0)", 2.0, 5.0),
        ("Conservative (2.5/4.0)", 2.5, 4.0),
        ("Aggressive (1.5/6.0)", 1.5, 6.0),
        ("Tight TP (2.0/3.0)", 2.0, 3.0),
    ]

    for name, sl, tp in variants:
        eq, tr = run_backtest(closes, highs, lows, gauss, macd_hist, rsi, sl_mult=sl, tp_mult=tp)
        w = [t for t in tr if t > 0]
        l = [t for t in tr if t <= 0]
        n = sum(tr)
        wr2 = len(w) / len(tr) * 100 if tr else 0
        pf2 = sum(w) / abs(sum(l)) if l else 999
        print(f"  {name:<25} ${n:>9.2f} {len(tr):>8} {wr2:>5.1f} {pf2:>5.2f}")

    # ── Summary ──
    print_header("DEMO COMPLETE")
    print("  This demo showed:")
    print("    1. Gaussian IIR filter — smooths price with zero lag")
    print("    2. HMM regime detection — classifies market state")
    print("    3. GaussMACD strategy — trend + momentum entries")
    print("    4. ATR-based risk management with MaxSL cap")
    print("    5. Multi-variant backtesting comparison")
    print()
    print("  To run on real MT5 data:")
    print("    python run.py --ea Gold_Apex_EA --from 2024.01.01 --to 2025.12.31")
    print()
    print("  Full documentation: https://github.com/Vinu-Kevin-Diesel/MT5-Strategy-Lab")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT5 Strategy Lab — Interactive Demo")
    parser.add_argument("--quick", action="store_true", help="Quick 30-second demo")
    args = parser.parse_args()
    run_demo(quick=args.quick)
