"""
商品期货盘口交易系统回测脚本
标的: ru2609 (天然橡胶期货)
数据: 逐笔tick盘口数据 (1.txt)

统计分析结论
=================
1. 每分钟约116个tick，1tick ≈ 0.52秒
2. tick级均值回归在手续费后期望为负（信号幅度 < 交易成本）
3. 1分钟K线 EMA(5/20)金叉/死叉策略接近平盈（-132元），但空头在上涨日全部亏损
4. 加入EMA(50)趋势过滤后（只顺主趋势方向交易），盈亏改善至+1352元，胜率50%

策略：1分钟K线 EMA三线趋势跟随
=========================================
  三条EMA：快线EMA(5)、慢线EMA(20)、趋势EMA(50)

  入场条件：
    做多：EMA5金叉EMA20（diff从负转正）AND diff>=2 AND 收盘价>EMA50（大趋势向上）
    做空：EMA5死叉EMA20（diff从正转负）AND diff<=-2 AND 收盘价<EMA50（大趋势向下）

  出场条件（优先级从高到低）：
    1. 止损 SL=25pt（即本K线High/Low触及止损线）
    2. 止盈 TP=60pt（即本K线High/Low触及止盈线）
    3. 反向信号（仅在趋势仍然不利于持仓方向时平仓）
    4. 收盘前5分钟强制平仓

  参数选择理由：
    - EMA5/20：适合1分钟周期的短中期交叉，5分钟 vs 20分钟趋势
    - EMA50：约50分钟的大趋势判断，有效过滤逆势交易
    - min_diff=2：避免EMA接近时的频繁小幅穿越（虚假信号）
    - TP=60pt：净盈594元/手，覆盖多次小止损
    - SL=25pt：净亏256元/手（含价差50元），保本胜率约30%
    - 实测胜率50% > 保本率30%，理论正期望

合约参数: 10吨/手, 最小变动5元/吨, 手续费单边3元/手
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ─── 参数 ─────────────────────────────────────────────────────────────────────
DATA_FILE = Path(__file__).parent / "1.txt"
CONTRACT_UNIT = 10          # 吨/手
TICK_SIZE = 5               # 元/吨（最小变动）
COMMISSION = 3              # 元/手（单边）

FAST_EMA = 5                # 快速EMA周期（分钟K线）
SLOW_EMA = 20               # 慢速EMA周期
TREND_EMA = 50              # 趋势过滤EMA周期

MIN_CROSS_DIFF = 2          # 元/吨：EMA5-EMA20差值绝对值须超过此值才视为有效穿越

TAKE_PROFIT = 60            # 止盈点数（元/吨）
STOP_LOSS = 25              # 止损点数（元/吨）

# 过滤掉的交易时段
LUNCH_START = "11:30"
LUNCH_END = "13:30"
NIGHT_START = "21:00"
NIGHT_END = "21:05"


# ─── 数据加载与K线构建 ───────────────────────────────────────────────────────
def load_ticks(filepath: Path) -> pd.DataFrame:
    """加载并预处理tick数据。"""
    df = pd.read_csv(filepath)
    df["dt"] = (
        pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    df = df[(df["LastPrice"] > 0) & (df["BidPrice1"] > 0)].copy()
    return df.reset_index(drop=True)


def build_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    """
    将tick数据重采样为1分钟OHLCV K线。

    保留首/末盘口价格用于模拟成交：
    - first_ask / first_bid: 本分钟第一个tick的价格（下根K线入场用）
    - last_bid / last_ask: 本分钟最后一个tick（平仓/收盘用）
    """
    idx_dt = ticks.set_index("dt")
    bars = idx_dt["LastPrice"].resample("1min").agg(
        open="first", high="max", low="min", close="last"
    )
    bars["first_ask"] = idx_dt["AskPrice1"].resample("1min").first()
    bars["first_bid"] = idx_dt["BidPrice1"].resample("1min").first()
    bars["last_bid"]  = idx_dt["BidPrice1"].resample("1min").last()
    bars["last_ask"]  = idx_dt["AskPrice1"].resample("1min").last()

    bars = bars.dropna(subset=["open", "close"])
    bars = bars[bars["close"] > 0]
    bars["time_str"] = bars.index.strftime("%H:%M")

    # 过滤午休、夜盘开盘及极早时段
    bars = bars[~((bars["time_str"] >= LUNCH_START) & (bars["time_str"] < LUNCH_END))]
    bars = bars[~((bars["time_str"] >= NIGHT_START) & (bars["time_str"] < NIGHT_END))]
    bars = bars[bars["time_str"] >= "09:05"]
    return bars.reset_index().rename(columns={"dt": "bar_time"})


def compute_signals(bars: pd.DataFrame) -> pd.DataFrame:
    """
    计算三条EMA及交叉信号。

    signal列: +1=做多信号, -1=做空信号, 0=无信号
    """
    c = bars["close"]
    bars["ema_fast"]  = c.ewm(span=FAST_EMA,  adjust=False).mean()
    bars["ema_slow"]  = c.ewm(span=SLOW_EMA,  adjust=False).mean()
    bars["ema_trend"] = c.ewm(span=TREND_EMA, adjust=False).mean()

    bars["ema_diff"]      = bars["ema_fast"] - bars["ema_slow"]
    bars["ema_diff_prev"] = bars["ema_diff"].shift(1)

    # 大趋势判断：收盘价在EMA50之上→上升趋势
    bars["trend_up"] = bars["close"] > bars["ema_trend"]

    bars["signal"] = 0

    # 做多：金叉 + 差值>=MIN_CROSS_DIFF + 大趋势向上
    long_mask = (
        (bars["ema_diff"] >= MIN_CROSS_DIFF) &
        (bars["ema_diff_prev"] < MIN_CROSS_DIFF) &
        bars["trend_up"]
    )
    # 做空：死叉 + 差值<=-MIN_CROSS_DIFF + 大趋势向下
    short_mask = (
        (bars["ema_diff"] <= -MIN_CROSS_DIFF) &
        (bars["ema_diff_prev"] > -MIN_CROSS_DIFF) &
        ~bars["trend_up"]
    )
    bars.loc[long_mask, "signal"] = 1
    bars.loc[short_mask, "signal"] = -1
    return bars


# ─── 回测引擎（K线级别）────────────────────────────────────────────────────────
def run_backtest(bars: pd.DataFrame) -> dict:
    """
    K线级别回测引擎。

    入场：信号K线收盘后，下一K线开盘（first_ask/first_bid）成交。
    平仓：优先检查本K线的High/Low是否触发SL/TP，再检查反向信号和收盘。

    SL/TP判断逻辑：假设在同一根K线内，SL先于TP触发（保守估计）。
    """
    position = None    # None 或 dict(direction, entry_price, entry_bar, entry_time)
    trades = []
    equity = []
    cumulative_pnl = 0.0
    n = len(bars)

    for i in range(n):
        row = bars.iloc[i]
        t_str = row["time_str"]
        force_close = ("14:55" <= t_str <= "15:00") or ("23:55" <= t_str)

        # ── 平仓检查 ──────────────────────────────────────────────────
        if position is not None:
            ep = position["entry_price"]
            direction = position["direction"]

            sl_price = (ep - STOP_LOSS)  if direction ==  1 else (ep + STOP_LOSS)
            tp_price = (ep + TAKE_PROFIT) if direction ==  1 else (ep - TAKE_PROFIT)

            close_reason = ""
            exit_price = None

            if force_close:
                # 收盘时以盘口对手价平仓
                exit_price = row["last_bid"] if direction == 1 else row["last_ask"]
                close_reason = "EOD"
            elif direction == 1:
                if row["low"] <= sl_price:       # SL先触发（保守假设）
                    exit_price = sl_price
                    close_reason = "SL"
                elif row["high"] >= tp_price:    # TP触发
                    exit_price = tp_price
                    close_reason = "TP"
            else:  # short
                if row["high"] >= sl_price:
                    exit_price = sl_price
                    close_reason = "SL"
                elif row["low"] <= tp_price:
                    exit_price = tp_price
                    close_reason = "TP"

            # 反向信号平仓（仅在趋势方向确认下）
            if not close_reason:
                sig = row["signal"]
                if direction == 1 and sig == -1:   # 持多，出现空头信号
                    exit_price = row["last_bid"]
                    close_reason = "REV"
                elif direction == -1 and sig == 1:  # 持空，出现多头信号
                    exit_price = row["last_ask"]
                    close_reason = "REV"

            if close_reason:
                pnl_pts = direction * (exit_price - ep)
                trade_pnl = pnl_pts * CONTRACT_UNIT - COMMISSION * 2
                cumulative_pnl += trade_pnl
                trades.append({
                    "entry_bar":   position["entry_bar"],
                    "exit_bar":    i,
                    "entry_time":  position["entry_time"],
                    "exit_time":   str(row["bar_time"]),
                    "entry_price": ep,
                    "exit_price":  exit_price,
                    "direction":   direction,
                    "pnl":         trade_pnl,
                    "reason":      close_reason,
                })
                position = None

        equity.append(cumulative_pnl)

        # ── 入场：若上一根K线有信号，本K线开盘入场 ───────────────────
        if i > 0 and position is None and not force_close:
            prev_signal = int(bars.iloc[i - 1]["signal"])
            if prev_signal != 0 and not (LUNCH_START <= t_str < LUNCH_END):
                direction = prev_signal
                ep = row["first_ask"] if direction == 1 else row["first_bid"]
                position = {
                    "direction":   direction,
                    "entry_price": ep,
                    "entry_bar":   i,
                    "entry_time":  str(row["bar_time"]),
                }

    # 尾部强制平仓
    if position is not None:
        last = bars.iloc[-1]
        ep = position["entry_price"]
        direction = position["direction"]
        exit_price = last["last_bid"] if direction == 1 else last["last_ask"]
        pnl_pts = direction * (exit_price - ep)
        trade_pnl = pnl_pts * CONTRACT_UNIT - COMMISSION * 2
        cumulative_pnl += trade_pnl
        trades.append({
            "entry_bar":   position["entry_bar"],
            "exit_bar":    n - 1,
            "entry_time":  position["entry_time"],
            "exit_time":   str(last["bar_time"]),
            "entry_price": ep,
            "exit_price":  exit_price,
            "direction":   direction,
            "pnl":         trade_pnl,
            "reason":      "END",
        })

    return {"trades": trades, "equity": equity, "final_pnl": cumulative_pnl}


# ─── 绩效统计 ─────────────────────────────────────────────────────────────────
def compute_stats(result: dict) -> dict:
    """计算常用回测绩效指标。"""
    trades = result["trades"]
    equity = result["equity"]
    if not trades:
        return {"error": "无交易记录"}

    df = pd.DataFrame(trades)
    pnls = df["pnl"].values
    eq = np.array(equity)

    total = len(df)
    winning = int((pnls > 0).sum())
    losing  = int((pnls <= 0).sum())
    gross_p = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    gross_l = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 0.0
    pf = gross_p / gross_l if gross_l > 0 else float("inf")

    peak  = np.maximum.accumulate(eq)
    max_dd = float((eq - peak).min())

    long_df  = df[df["direction"] ==  1]
    short_df = df[df["direction"] == -1]

    return {
        "总交易次数":     total,
        "多头次数":       int(len(long_df)),
        "空头次数":       int(len(short_df)),
        "盈利次数":       winning,
        "亏损次数":       losing,
        "胜率":          f"{winning / total:.2%}",
        "总盈亏(元)":    round(result["final_pnl"], 2),
        "多头盈亏(元)":  round(float(long_df["pnl"].sum()),  2),
        "空头盈亏(元)":  round(float(short_df["pnl"].sum()), 2),
        "毛利润(元)":    round(gross_p, 2),
        "毛亏损(元)":    round(gross_l, 2),
        "盈亏比":        round(pf, 3),
        "平均盈利(元)":  round(float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0, 2),
        "平均亏损(元)":  round(float(abs(pnls[pnls < 0].mean())) if (pnls < 0).any() else 0, 2),
        "最大回撤(元)":  round(max_dd, 2),
        "平仓原因":      df["reason"].value_counts().to_dict(),
    }


# ─── 可视化 ───────────────────────────────────────────────────────────────────
def plot_results(bars: pd.DataFrame, result: dict) -> None:
    """绘制K线价格+三EMA、净值曲线、EMA差值图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("提示: 未安装 matplotlib，跳过图表绘制。")
        return

    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    trades_df = pd.DataFrame(result["trades"]) if result["trades"] else pd.DataFrame()
    equity = result["equity"]
    idx = range(len(bars))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 子图1：价格 + 三EMA + 信号
    ax1 = axes[0]
    ax1.plot(idx, bars["close"].values,     color="steelblue", lw=0.7, label="1min Close")
    ax1.plot(idx, bars["ema_fast"].values,  color="orange",    lw=1.0, label=f"EMA{FAST_EMA}")
    ax1.plot(idx, bars["ema_slow"].values,  color="red",       lw=1.2, label=f"EMA{SLOW_EMA}")
    ax1.plot(idx, bars["ema_trend"].values, color="purple",    lw=1.5, linestyle="--",
             label=f"EMA{TREND_EMA}(趋势)")
    if not trades_df.empty:
        longs  = trades_df[trades_df["direction"] ==  1]
        shorts = trades_df[trades_df["direction"] == -1]
        if len(longs):
            li = longs["entry_bar"].values
            ax1.scatter(li, bars.iloc[li]["close"].values,
                        marker="^", color="red",   s=80, zorder=5, label="做多入场")
        if len(shorts):
            si = shorts["entry_bar"].values
            ax1.scatter(si, bars.iloc[si]["close"].values,
                        marker="v", color="green", s=80, zorder=5, label="做空入场")
    ax1.set_title(
        f"ru2609 1分钟K线 EMA{FAST_EMA}/{SLOW_EMA}/{TREND_EMA} 三线趋势策略"
    )
    ax1.set_ylabel("价格（元/吨）")
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    # 子图2：净值曲线
    ax2 = axes[1]
    ax2.plot(equity, color="darkorange", lw=1, label="累计盈亏(元)")
    ax2.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax2.fill_between(range(len(equity)), equity, 0,
                     where=[e >= 0 for e in equity], alpha=0.3, color="green")
    ax2.fill_between(range(len(equity)), equity, 0,
                     where=[e < 0 for e in equity], alpha=0.3, color="red")
    ax2.set_title("累计盈亏净值曲线")
    ax2.set_ylabel("盈亏（元）")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 子图3：EMA差值
    ax3 = axes[2]
    diff = bars["ema_diff"].values
    ax3.plot(idx, diff, color="navy", lw=0.8, label=f"EMA差值(EMA{FAST_EMA}-EMA{SLOW_EMA})")
    ax3.axhline(0,              color="gray",  lw=0.8)
    ax3.axhline( MIN_CROSS_DIFF, color="green", lw=0.8, linestyle="--",
                label=f"有效穿越阈值 ±{MIN_CROSS_DIFF}")
    ax3.axhline(-MIN_CROSS_DIFF, color="red",   lw=0.8, linestyle="--")
    ax3.fill_between(idx, diff, 0, where=diff >= 0, alpha=0.2, color="green", label="快>慢（偏多）")
    ax3.fill_between(idx, diff, 0, where=diff <  0, alpha=0.2, color="red",   label="快<慢（偏空）")
    ax3.set_title(f"EMA趋势强度（EMA{FAST_EMA} - EMA{SLOW_EMA}）")
    ax3.set_ylabel("差值（元/吨）")
    ax3.set_xlabel("1分钟K线序号")
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / "backtest_result.png"
    plt.savefig(out_path, dpi=150)
    print(f"图表已保存至: {out_path}")
    plt.close()


# ─── 主程序 ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" ru2609 天然橡胶「1分钟EMA三线趋势跟随」策略 回测")
    print("=" * 60)
    print()

    print(f"加载数据: {DATA_FILE}")
    ticks = load_ticks(DATA_FILE)
    print(f"有效tick数: {len(ticks)}")

    bars = build_bars(ticks)
    print(f"1分钟K线数: {len(bars)}")
    print(f"价格范围: {ticks['LastPrice'].min()} ~ {ticks['LastPrice'].max()} 元/吨")
    print(f"开盘: {ticks['LastPrice'].iloc[0]}  收盘: {ticks['LastPrice'].iloc[-1]}")
    print()

    bars = compute_signals(bars)
    n_long  = (bars["signal"] ==  1).sum()
    n_short = (bars["signal"] == -1).sum()
    print(f"做多信号: {n_long} 次，做空信号: {n_short} 次")
    print()

    print("运行回测...")
    result = run_backtest(bars)
    print(f"共产生 {len(result['trades'])} 笔交易")
    print()

    stats = compute_stats(result)
    print("── 回测绩效统计 ──────────────────────────────")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()

    if result["trades"]:
        trades_df = pd.DataFrame(result["trades"])
        trades_df["dir_str"] = trades_df["direction"].map({1: "多", -1: "空"})
        print("── 交易明细 ─────────────────────────────────")
        print(trades_df[["entry_time", "exit_time", "entry_price", "exit_price",
                          "dir_str", "pnl", "reason"]].to_string(index=False))
        print()
        out_csv = Path(__file__).parent / "trades.csv"
        trades_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"完整交易记录已保存至: {out_csv}")

    plot_results(bars, result)

    print()
    print("策略说明")
    print("─" * 40)
    net_tp = TAKE_PROFIT * CONTRACT_UNIT - COMMISSION * 2
    net_sl = STOP_LOSS * CONTRACT_UNIT + COMMISSION * 2
    print(f"  策略类型    : 1分钟K线 EMA三线趋势跟随")
    print(f"  信号计算    : EMA{FAST_EMA}（快）vs EMA{SLOW_EMA}（慢）vs EMA{TREND_EMA}（趋势）")
    print(f"  入场条件    : EMA金叉/死叉 AND |差值|>={MIN_CROSS_DIFF}pt AND 价格顺EMA{TREND_EMA}方向")
    print(f"  出场条件    : 止盈{TAKE_PROFIT}pt / 止损{STOP_LOSS}pt / 反向信号 / 收盘")
    print(f"  止盈        : {TAKE_PROFIT}pt → 净盈利约{net_tp}元/手")
    print(f"  止损        : {STOP_LOSS}pt → 净亏损约{net_sl+10*TICK_SIZE}元/手（含价差）")
    print(f"  理论保本率  : {net_sl/(net_tp+net_sl)*100:.1f}%")
    print(f"  单边手续费  : {COMMISSION}元/手")
    print(f"  合约乘数    : {CONTRACT_UNIT}吨/手，最小变动{TICK_SIZE}元/吨")


if __name__ == "__main__":
    main()
