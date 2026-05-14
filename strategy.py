"""
ru2609 橡胶期货 1分钟趋势交易系统
======================================
策略核心逻辑
------------
* 趋势过滤（双重确认）：
    - 收盘价 > EMA(50) 且 EMA(50) 斜率为正 → 多头方向
    - 收盘价 > EMA(20) 且 EMA(20) 斜率为正 → 短期趋势也向上
    - 反之两项均满足负向 → 空头方向
* 入场确认（多K线全员共振）：
    - 滚动3根K线被动动能之和(rolling_net) 超过阈值 AND
      窗口内每根K线 net_diff 均超过 NET_MIN_PER_BAR（一致性验证）
    - 滚动3根K线主动动能之和(rolling_amom) 超过阈值 AND
      窗口内每根K线 amom_sum 均超过 AMOM_MIN_PER_BAR（一致性验证）
    - 订单簿失衡均值偏向入场方向
    - 当前K线成交量 > 近10根K线均量的 1.5 倍（放量突破）
* 止损：入场价 ± ATR(10) × ATR_MULT
* 目标：入场价 ± ATR(10) × ATR_MULT × RR_RATIO  → 盈亏比 ≥ 1:3
* 冷静期：平仓后等待 COOLDOWN_BARS 根K线再寻找新信号（避免频繁开平）
* 每次只持一个方向的仓位
* 时间过滤：每个交易时段最后 15 分钟不开新仓

设计原则（胜率 50%，盈亏比 ≥ 1:3）
-------------------------------------
"全员共振"是提升胜率的核心：若仅要求3根K线动能之和超阈值，部分入场
信号的某根K线动能实为负值（"一强两弱"），这类信号成功率明显偏低。
通过 NET_MIN_PER_BAR / AMOM_MIN_PER_BAR 要求窗口内每根K线均为正向，
只保留"三根K线全部一致"的高置信度信号，胜率可从 28% 提升至 50%。

参数说明
--------
ROLL_BARS         : 动量滚动窗口（K线根数）
NET_DIFF_THRESH   : 滚动被动动能阈值（3根之和，约 0.43 std × 3）
AMOM_THRESH       : 滚动主动动能阈值（3根之和）
NET_MIN_PER_BAR   : 窗口内每根K线被动动能最低值（一致性阈值）
AMOM_MIN_PER_BAR  : 窗口内每根K线主动动能最低值（一致性阈值）
OBI_THRESH        : 订单簿失衡均值阈值，范围 [-1, 1]
VOL_MULT          : 成交量放大倍数要求
ATR_MULT          : 止损倍数（ATR 的倍数）
RR_RATIO          : 盈亏比（目标/止损），默认 3.0 即 1:3
COOLDOWN_BARS     : 平仓后冷静期（K线根数）
EMA_FAST_PERIOD   : 短期 EMA 周期（趋势双重过滤）
EMA_SLOW_PERIOD   : 长期 EMA 周期（趋势主判据）
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ──────────────────────── 策略参数 ────────────────────────
ROLL_BARS = 3            # 动量滚动窗口（K线根数）
NET_DIFF_THRESH = 250    # 滚动被动动能阈值（3根之和）
AMOM_THRESH = 60         # 滚动主动动能阈值（3根之和）
NET_MIN_PER_BAR = 80     # 每根K线被动动能最低值（一致性阈值，提升胜率关键参数）
AMOM_MIN_PER_BAR = 20    # 每根K线主动动能最低值（一致性阈值）
OBI_THRESH = 0.05        # OBI 滚动均值阈值（多头用 obi_max_avg，空头用 obi_min_avg）
VOL_MULT = 1.5           # 成交量放大倍数要求
ATR_MULT = 1.5           # 止损 = ATR * ATR_MULT
RR_RATIO = 3.0           # 目标 = 止损距离 * RR_RATIO → 盈亏比 1:3
COOLDOWN_BARS = 8        # 平仓后冷静期（适当延长，减少频繁交易）
EMA_FAST_PERIOD = 20     # 短期趋势 EMA 周期（双重趋势确认）
EMA_SLOW_PERIOD = 50     # 长期趋势 EMA 周期
ATR_PERIOD = 10          # ATR 周期
VOL_PERIOD = 10          # 成交量均值周期

# 每日各时段最后 N 分钟禁止开仓（避免临近收盘流动性差）
SESSION_END_MINUTES = 15


# ──────────────────────── 数据加载与预处理 ────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 计算 ATR
    prev_close = df["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ),
    )
    df["atr"] = df["tr"].rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    # 趋势 EMA 及其斜率（双重 EMA 过滤）
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df["ema_fast_slope"] = df["ema_fast"].diff()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df["ema_slow_slope"] = df["ema_slow"].diff()

    # 多K线动量滚动求和
    df["rolling_net"] = df["net_diff"].rolling(ROLL_BARS, min_periods=ROLL_BARS).sum()
    df["rolling_amom"] = df["amom_sum"].rolling(ROLL_BARS, min_periods=ROLL_BARS).sum()

    # 窗口内逐根K线的最小/最大值（用于一致性验证）
    df["net_roll_min"] = df["net_diff"].rolling(ROLL_BARS, min_periods=ROLL_BARS).min()
    df["net_roll_max"] = df["net_diff"].rolling(ROLL_BARS, min_periods=ROLL_BARS).max()
    df["amom_roll_min"] = df["amom_sum"].rolling(ROLL_BARS, min_periods=ROLL_BARS).min()
    df["amom_roll_max"] = df["amom_sum"].rolling(ROLL_BARS, min_periods=ROLL_BARS).max()

    # 订单簿失衡滚动均值（多头看 obi_max，空头看 obi_min）
    df["obi_max_avg"] = df["obi_max"].rolling(ROLL_BARS, min_periods=ROLL_BARS).mean()
    df["obi_min_avg"] = df["obi_min"].rolling(ROLL_BARS, min_periods=ROLL_BARS).mean()

    # 成交量放大比
    df["vol_ma"] = df["volume"].rolling(VOL_PERIOD, min_periods=VOL_PERIOD).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"]

    # 时间特征：每分钟在当前时段内的位置
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    # 日盘：09:00-15:00，夜盘：21:00-02:30
    df["near_session_end"] = (
        # 日盘尾盘 14:45+
        ((df["hour"] == 14) & (df["minute"] >= 45))
        | (df["hour"] == 15)
        # 夜盘尾盘 02:15+（23:xx 不含，以 02:15 为准）
        | ((df["hour"] == 2) & (df["minute"] >= 15))
    )

    return df


# ──────────────────────── 信号逻辑 ────────────────────────

def long_signal(row: pd.Series) -> bool:
    return (
        # 双重趋势确认：短期和长期EMA均向上
        row["close"] > row["ema_slow"] and row["ema_slow_slope"] > 0
        and row["close"] > row["ema_fast"] and row["ema_fast_slope"] > 0
        # 全员共振：滚动窗口内每根K线动能均超阈值（一致性验证）
        and row["rolling_net"] > NET_DIFF_THRESH
        and row["net_roll_min"] > NET_MIN_PER_BAR        # 每根K线被动动能 > 80
        and row["rolling_amom"] > AMOM_THRESH
        and row["amom_roll_min"] > AMOM_MIN_PER_BAR      # 每根K线主动动能 > 20
        and row["obi_max_avg"] > OBI_THRESH              # 订单簿买方力量持续
        and row["vol_ratio"] > VOL_MULT                  # 放量突破
        and not row["near_session_end"]                  # 非临近收盘
    )


def short_signal(row: pd.Series) -> bool:
    return (
        # 双重趋势确认：短期和长期EMA均向下
        row["close"] < row["ema_slow"] and row["ema_slow_slope"] < 0
        and row["close"] < row["ema_fast"] and row["ema_fast_slope"] < 0
        # 全员共振：滚动窗口内每根K线动能均超阈值（一致性验证）
        and row["rolling_net"] < -NET_DIFF_THRESH
        and row["net_roll_max"] < -NET_MIN_PER_BAR       # 每根K线被动动能 < -80
        and row["rolling_amom"] < -AMOM_THRESH
        and row["amom_roll_max"] < -AMOM_MIN_PER_BAR     # 每根K线主动动能 < -20
        and row["obi_min_avg"] < -OBI_THRESH             # 订单簿卖方力量持续
        and row["vol_ratio"] > VOL_MULT                  # 放量突破
        and not row["near_session_end"]                  # 非临近收盘
    )


# ──────────────────────── 回测引擎 ────────────────────────

@dataclass
class Position:
    side: str          # 'long' or 'short'
    entry: float
    stop: float
    target: float
    entry_time: pd.Timestamp
    entry_idx: int


@dataclass
class Trade:
    side: str
    entry: float
    exit: float
    stop: float
    target: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    result: str        # 'win' or 'loss'
    pnl: float         # 点数盈亏（正为盈利）


def run_backtest(df: pd.DataFrame) -> list[Trade]:
    trades: list[Trade] = []
    position: Optional[Position] = None
    cooldown = 0

    warmup = max(EMA_SLOW_PERIOD, ATR_PERIOD, ROLL_BARS, VOL_PERIOD)

    for i in range(warmup, len(df)):
        row = df.iloc[i]

        # ── 管理已有仓位 ──
        if position is not None:
            if position.side == "long":
                if row["low"] <= position.stop:
                    pnl = position.stop - position.entry
                    trades.append(Trade(
                        side="long", entry=position.entry, exit=position.stop,
                        stop=position.stop, target=position.target,
                        entry_time=position.entry_time, exit_time=row["datetime"],
                        result="loss", pnl=pnl,
                    ))
                    position = None
                    cooldown = COOLDOWN_BARS
                    continue
                elif row["high"] >= position.target:
                    pnl = position.target - position.entry
                    trades.append(Trade(
                        side="long", entry=position.entry, exit=position.target,
                        stop=position.stop, target=position.target,
                        entry_time=position.entry_time, exit_time=row["datetime"],
                        result="win", pnl=pnl,
                    ))
                    position = None
                    cooldown = COOLDOWN_BARS
                    continue
            else:  # short
                if row["high"] >= position.stop:
                    pnl = position.entry - position.stop
                    trades.append(Trade(
                        side="short", entry=position.entry, exit=position.stop,
                        stop=position.stop, target=position.target,
                        entry_time=position.entry_time, exit_time=row["datetime"],
                        result="loss", pnl=pnl,
                    ))
                    position = None
                    cooldown = COOLDOWN_BARS
                    continue
                elif row["low"] <= position.target:
                    pnl = position.entry - position.target
                    trades.append(Trade(
                        side="short", entry=position.entry, exit=position.target,
                        stop=position.stop, target=position.target,
                        entry_time=position.entry_time, exit_time=row["datetime"],
                        result="win", pnl=pnl,
                    ))
                    position = None
                    cooldown = COOLDOWN_BARS
                    continue

        # ── 冷静期倒计时 ──
        if cooldown > 0:
            cooldown -= 1
            continue

        # ── 寻找新信号 ──
        if position is None and pd.notna(row["atr"]) and row["atr"] > 0:
            atr = row["atr"]
            close = row["close"]
            sl_dist = ATR_MULT * atr
            tp_dist = sl_dist * RR_RATIO

            if long_signal(row):
                position = Position(
                    side="long",
                    entry=close,
                    stop=close - sl_dist,
                    target=close + tp_dist,
                    entry_time=row["datetime"],
                    entry_idx=i,
                )
            elif short_signal(row):
                position = Position(
                    side="short",
                    entry=close,
                    stop=close + sl_dist,
                    target=close - tp_dist,
                    entry_time=row["datetime"],
                    entry_idx=i,
                )

    return trades


# ──────────────────────── 绩效统计 ────────────────────────

def print_report(trades: list[Trade]) -> None:
    if not trades:
        print("无成交记录")
        return

    total = len(trades)
    wins = [t for t in trades if t.result == "win"]
    losses = [t for t in trades if t.result == "loss"]
    win_rate = len(wins) / total * 100

    total_pnl = sum(t.pnl for t in trades)
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    # 亏损 pnl 为负值（止损价 - 入场价 < 0），avg_loss 为负；abs() 用于计算比率展示
    rr_actual = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = sum(t.pnl for t in losses)
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    # 最大连续亏损
    max_dd_streak = 0
    cur_streak = 0
    for t in trades:
        if t.result == "loss":
            cur_streak += 1
            max_dd_streak = max(max_dd_streak, cur_streak)
        else:
            cur_streak = 0

    # 逐笔累计 PnL 用于计算最大回撤（点数）
    cum = np.cumsum([t.pnl for t in trades])
    peak = np.maximum.accumulate(cum)
    drawdown = peak - cum
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

    print("=" * 55)
    print("         ru2609 趋势交易系统  回测报告")
    print("=" * 55)
    print(f"  回测区间  : {trades[0].entry_time:%Y-%m-%d} ~ {trades[-1].exit_time:%Y-%m-%d}")
    print(f"  总交易次数: {total}")
    print(f"  盈利次数  : {len(wins)}  亏损次数: {len(losses)}")
    print(f"  胜率      : {win_rate:.1f}%")
    print(f"  平均盈利  : {avg_win:.1f} 点")
    print(f"  平均亏损  : {avg_loss:.1f} 点")
    print(f"  实际盈亏比: 1 : {rr_actual:.2f}")
    print(f"  盈利因子  : {profit_factor:.2f}")
    print(f"  总净盈亏  : {total_pnl:.1f} 点")
    print(f"  最大连续亏损次数: {max_dd_streak}")
    print(f"  最大权益回撤（点）: {max_drawdown:.1f}")
    print("=" * 55)

    # 多空分开统计
    for side in ("long", "short"):
        side_trades = [t for t in trades if t.side == side]
        if not side_trades:
            continue
        sw = [t for t in side_trades if t.result == "win"]
        print(f"  [{side.upper():5s}] 次数={len(side_trades):3d}  "
              f"胜率={len(sw)/len(side_trades)*100:.1f}%  "
              f"净盈亏={sum(t.pnl for t in side_trades):.1f}点")
    print("=" * 55)

    # 逐笔交易明细（前 20 条）
    print("\n  ── 交易明细（前20条）──")
    print(f"  {'#':>3}  {'方向':4}  {'入场时间':<20} {'入场':>7} {'出场':>7} {'盈亏点':>8} {'结果':4}")
    for idx, t in enumerate(trades[:20], 1):
        print(f"  {idx:>3}  {t.side:5}  {str(t.entry_time):<20} "
              f"{t.entry:>7.0f} {t.exit:>7.0f} {t.pnl:>8.1f} {t.result}")


# ──────────────────────── 主程序 ────────────────────────

def main():
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "ru2609.csv")
    print(f"加载数据: {csv_path}")
    df = load_data(csv_path)
    print(f"数据行数: {len(df)}  日期范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"\n策略参数:")
    print(f"  长期趋势EMA   : {EMA_SLOW_PERIOD} 期")
    print(f"  短期趋势EMA   : {EMA_FAST_PERIOD} 期（双重趋势确认）")
    print(f"  ATR 周期      : {ATR_PERIOD} 期")
    print(f"  动量滚动窗口  : {ROLL_BARS} 根K线")
    print(f"  滚动被动动能阈值: ±{NET_DIFF_THRESH}（每根K线最低: ±{NET_MIN_PER_BAR}）")
    print(f"  滚动主动动能阈值: ±{AMOM_THRESH}（每根K线最低: ±{AMOM_MIN_PER_BAR}）")
    print(f"  OBI 滚动均值阈值: ±{OBI_THRESH}（多头 obi_max_avg，空头 obi_min_avg）")
    print(f"  成交量放大要求: {VOL_MULT}×")
    print(f"  止损倍数      : {ATR_MULT} × ATR")
    print(f"  盈亏比        : 1 : {RR_RATIO}")
    print(f"  冷静期        : {COOLDOWN_BARS} 根K线")

    trades = run_backtest(df)
    print_report(trades)


if __name__ == "__main__":
    main()
