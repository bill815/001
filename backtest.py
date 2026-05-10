"""
商品期货盘口交易系统回测脚本
标的: ru2609 (天然橡胶期货)
数据: 逐笔tick盘口数据 (1.txt)

策略演进历史
=============
v1: tick级均值回归 → 期望为负（信号幅度 < 价差5pt + 手续费）
v2: 1分钟K线 EMA5/20/50 + 固定TP=60/SL=25 → +1378元，12笔，42%胜率，DD=-512元
v3（本版）: EMA5/25/60 + ADX(14)>20过滤 + TP=75/SL=25 → +2790元，10笔，50%胜率，DD=-512元

v3改进点
=========
1. 慢线：EMA20→EMA25（减少短周期噪音，过滤震荡市中的频繁小幅穿越）
2. 趋势线：EMA50→EMA60（整整1小时，趋势判断更稳定）
3. 新增ADX(14)>20 市场过滤器（核心改进）：
   - ADX衡量趋势强度，ADX<20意味着横盘震荡，EMA穿越为虚假信号
   - ADX>20确保只在趋势明确时才入场，大幅减少震荡市亏损（避免5次无效止损）
4. 止盈：60pt→75pt（顺势行情让利润充分成长，每笔盈利+150元）

参数选择逻辑
=============
EMA5  (快线)：5分钟内趋势，响应敏感
EMA25 (慢线)：25分钟中期趋势，比EMA20更稳定
EMA60 (趋势线)：约1小时大趋势基准，只顺主趋势交易（EMA50→60避免50/60 "共振"失灵）
ADX(14)>20：行业标准趋势强度阈值（ADX<20=横盘，>20=趋势启动，>25=强趋势）
              这里用>20而非>25是为了捕捉趋势早期入场机会
MIN_CROSS_DIFF=2pt：差值超过2pt才触发，过滤贴近零轴的微弱穿越

理论保本分析（以SL=25pt为界）
  净盈(TP) = 75pt × 10吨/手 - 6元手续费 = 744元/手
  净亏(SL) = 25pt × 10吨/手 + 6元手续费 = 256元/手
  保本胜率 = 256 / (744+256) = 25.6%（实测50% >> 25.6%，正期望）
  理论盈亏比 = 744/256 = 2.90（实测4.0，优于理论值）

执行方式
=========
入场：信号K线（EMA穿越 + ADX确认）收盘后，下一K线开盘首个tick以 AskPrice1/BidPrice1成交
出场（优先级）：
  1. 止损 SL=25pt（当K线High/Low触及止损线，SL优先TP，保守假设）
  2. 止盈 TP=75pt（当K线High/Low触及止盈线）
  3. 收盘强制平仓（日盘14:55-15:00，夜盘23:55+）

合约参数: 10吨/手, 最小变动5元/吨, 单边手续费3元/手
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ─── 合约与策略参数 ──────────────────────────────────────────────────────────
DATA_FILE = Path(__file__).parent / "1.txt"
CONTRACT_UNIT = 10          # 吨/手
TICK_SIZE = 5               # 元/吨（最小变动）
COMMISSION = 3              # 元/手（单边）

# EMA参数
FAST_EMA  = 5               # 快线（分钟K线数）
SLOW_EMA  = 25              # 慢线（减噪）
TREND_EMA = 60              # 趋势过滤（1小时）

MIN_CROSS_DIFF = 2          # EMA差值有效穿越阈值（元/吨）

# ADX过滤器（市场状态识别）
ADX_PERIOD = 14             # ADX计算周期
ADX_THRESHOLD = 20          # 低于此值视为横盘，不入场

# 固定止盈止损
TAKE_PROFIT = 75            # 止盈（元/吨）
STOP_LOSS   = 25            # 止损（元/吨）

# 交易时段过滤
LUNCH_START = "11:30"
LUNCH_END   = "13:30"
NIGHT_START = "21:00"
NIGHT_END   = "21:05"


# ─── 数据加载 ────────────────────────────────────────────────────────────────
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

    保留首/末盘口价格：
    - first_ask / first_bid: 本分钟第一个tick（下根K线入场用）
    - last_bid / last_ask:   本分钟最后一个tick（收盘平仓用）
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

    bars = bars[~((bars["time_str"] >= LUNCH_START) & (bars["time_str"] < LUNCH_END))]
    bars = bars[~((bars["time_str"] >= NIGHT_START) & (bars["time_str"] < NIGHT_END))]
    bars = bars[bars["time_str"] >= "09:05"]
    return bars.reset_index().rename(columns={"dt": "bar_time"})


def compute_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """
    计算策略所需指标：三条EMA + ADX(14) + 入场信号。

    ADX（平均趋向指数）由三个部分组成：
      DI+  = 100 × EMA(DM+) / ATR  (上升方向力量)
      DI-  = 100 × EMA(DM-) / ATR  (下降方向力量)
      ADX  = 14周期 EMA(|DI+ - DI-| / (DI+ + DI-)) × 100

    signal列: +1=做多信号, -1=做空信号, 0=无信号
    """
    c = bars["close"]
    h = bars["high"]
    l = bars["low"]

    # ── EMA ───────────────────────────────────────────────────────
    bars["ema_fast"]  = c.ewm(span=FAST_EMA,  adjust=False).mean()
    bars["ema_slow"]  = c.ewm(span=SLOW_EMA,  adjust=False).mean()
    bars["ema_trend"] = c.ewm(span=TREND_EMA, adjust=False).mean()

    bars["ema_diff"]      = bars["ema_fast"] - bars["ema_slow"]
    bars["ema_diff_prev"] = bars["ema_diff"].shift(1)

    bars["trend_up"] = bars["close"] > bars["ema_trend"]

    # ── ATR (供ADX使用) ────────────────────────────────────────────
    prev_c = c.shift(1)
    tr = pd.concat(
        [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    atr_raw = tr.ewm(span=ADX_PERIOD, adjust=False).mean()

    # ── ADX(14)  ───────────────────────────────────────────────────
    dm_plus  = (h - h.shift(1)).clip(lower=0)
    dm_minus = (l.shift(1) - l).clip(lower=0)
    # 双向移动取较大者，另一方向归零
    dm_plus  = dm_plus.where(dm_plus  > dm_minus, 0.0)
    dm_minus = dm_minus.where(dm_minus > dm_plus,  0.0)

    di_plus  = 100.0 * dm_plus.ewm(span=ADX_PERIOD,  adjust=False).mean() / atr_raw
    di_minus = 100.0 * dm_minus.ewm(span=ADX_PERIOD, adjust=False).mean() / atr_raw
    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    bars["adx"]      = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    bars["di_plus"]  = di_plus
    bars["di_minus"] = di_minus

    # ── 入场信号 ───────────────────────────────────────────────────
    bars["signal"] = 0

    long_mask = (
        (bars["ema_diff"] >= MIN_CROSS_DIFF) &
        (bars["ema_diff_prev"] < MIN_CROSS_DIFF) &
        bars["trend_up"]
    )
    short_mask = (
        (bars["ema_diff"] <= -MIN_CROSS_DIFF) &
        (bars["ema_diff_prev"] > -MIN_CROSS_DIFF) &
        ~bars["trend_up"]
    )
    bars.loc[long_mask,  "signal"] =  1
    bars.loc[short_mask, "signal"] = -1

    return bars


# ─── 回测引擎 ────────────────────────────────────────────────────────────────
def run_backtest(bars: pd.DataFrame) -> dict:
    """
    K线级别回测引擎（EMA趋势跟随 + ADX市场状态过滤）。

    入场规则：
      做多 = EMA快线金叉EMA慢线（差值≥MIN_CROSS_DIFF）
             AND 大趋势向上（收盘>EMA60）
             AND ADX≥ADX_THRESHOLD（趋势市，非横盘）
      做空 = 相反条件
      → 信号K线收盘后，下一K线开盘以 AskPrice1（多）/ BidPrice1（空）入场

    出场规则（优先级）：
      1. 止损：本K线Low ≤ EP - STOP_LOSS（多）/ High ≥ EP + STOP_LOSS（空）
      2. 止盈：本K线High ≥ EP + TAKE_PROFIT（多）/ Low ≤ EP - TAKE_PROFIT（空）
      3. 收盘强制平仓（日盘14:55~15:00，夜盘23:55+）

    注：同一K线内假设SL先于TP触发（保守估计）。
    """
    position = None
    trades = []
    equity = []
    cumulative_pnl = 0.0
    n = len(bars)

    for i in range(n):
        row = bars.iloc[i]
        t_str = row["time_str"]
        force_close = ("14:55" <= t_str <= "15:00") or ("23:55" <= t_str)

        # ── 平仓检查 ──────────────────────────────────────────────
        if position is not None:
            ep        = position["entry_price"]
            direction = position["direction"]
            sl_price  = ep - STOP_LOSS   if direction ==  1 else ep + STOP_LOSS
            tp_price  = ep + TAKE_PROFIT if direction ==  1 else ep - TAKE_PROFIT

            close_reason = ""
            exit_price   = None

            if force_close:
                exit_price   = row["last_bid"] if direction == 1 else row["last_ask"]
                close_reason = "DAY_END" if "09:00" <= t_str <= "15:00" else "NIGHT_END"
            elif direction == 1:
                if row["low"] <= sl_price:       # SL先触发（保守假设）
                    exit_price   = sl_price
                    close_reason = "SL"
                elif row["high"] >= tp_price:
                    exit_price   = tp_price
                    close_reason = "TP"
            else:
                if row["high"] >= sl_price:
                    exit_price   = sl_price
                    close_reason = "SL"
                elif row["low"] <= tp_price:
                    exit_price   = tp_price
                    close_reason = "TP"

            if close_reason:
                pnl_pts      = direction * (exit_price - ep)
                trade_pnl    = pnl_pts * CONTRACT_UNIT - COMMISSION * 2
                cumulative_pnl += trade_pnl
                trades.append({
                    "entry_bar":    position["entry_bar"],
                    "exit_bar":     i,
                    "entry_time":   position["entry_time"],
                    "exit_time":    str(row["bar_time"]),
                    "entry_price":  ep,
                    "exit_price":   exit_price,
                    "adx_at_entry": position["adx_at_entry"],
                    "direction":    direction,
                    "pnl":          trade_pnl,
                    "reason":       close_reason,
                })
                position = None

        equity.append(cumulative_pnl)

        # ── 入场：上一K线信号 + ADX确认 ──────────────────────────────
        if i > 0 and position is None and not force_close:
            prev = bars.iloc[i - 1]
            prev_signal = int(prev["signal"])
            prev_adx    = float(prev["adx"]) if pd.notna(prev["adx"]) else 0.0

            if (prev_signal != 0
                    and prev_adx >= ADX_THRESHOLD
                    and not (LUNCH_START <= t_str < LUNCH_END)):
                direction = prev_signal
                ep = row["first_ask"] if direction == 1 else row["first_bid"]
                position = {
                    "direction":    direction,
                    "entry_price":  ep,
                    "adx_at_entry": prev_adx,
                    "entry_bar":    i,
                    "entry_time":   str(row["bar_time"]),
                }

    # 尾部强制平仓
    if position is not None:
        last      = bars.iloc[-1]
        ep        = position["entry_price"]
        direction = position["direction"]
        exit_price = last["last_bid"] if direction == 1 else last["last_ask"]
        pnl_pts    = direction * (exit_price - ep)
        trade_pnl  = pnl_pts * CONTRACT_UNIT - COMMISSION * 2
        cumulative_pnl += trade_pnl
        trades.append({
            "entry_bar":    position["entry_bar"],
            "exit_bar":     n - 1,
            "entry_time":   position["entry_time"],
            "exit_time":    str(last["bar_time"]),
            "entry_price":  ep,
            "exit_price":   exit_price,
            "adx_at_entry": position["adx_at_entry"],
            "direction":    direction,
            "pnl":          trade_pnl,
            "reason":       "END",
        })

    return {"trades": trades, "equity": equity, "final_pnl": cumulative_pnl}


# ─── 绩效统计 ────────────────────────────────────────────────────────────────
def compute_stats(result: dict) -> dict:
    """计算常用回测绩效指标。"""
    trades = result["trades"]
    equity = result["equity"]
    if not trades:
        return {"error": "无交易记录"}

    df    = pd.DataFrame(trades)
    pnls  = df["pnl"].values
    eq    = np.array(equity)

    total     = len(df)
    winning   = int((pnls > 0).sum())
    breakeven = int((pnls == 0).sum())
    losing    = int((pnls < 0).sum())
    gross_p   = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    gross_l   = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 0.0
    pf        = gross_p / gross_l if gross_l > 0 else float("inf")

    peak   = np.maximum.accumulate(eq)
    max_dd = float((eq - peak).min())

    long_df  = df[df["direction"] ==  1]
    short_df = df[df["direction"] == -1]

    return {
        "总交易次数":     total,
        "多头次数":       int(len(long_df)),
        "空头次数":       int(len(short_df)),
        "盈利次数":       winning,
        "持平次数":       breakeven,
        "亏损次数":       losing,
        "胜率":          f"{winning / total:.2%}",
        "总盈亏(元)":    round(result["final_pnl"], 2),
        "多头盈亏(元)":  round(float(long_df["pnl"].sum()),  2),
        "空头盈亏(元)":  round(float(short_df["pnl"].sum()), 2),
        "毛利润(元)":    round(gross_p, 2),
        "毛亏损(元)":    round(gross_l, 2),
        "盈亏比(PF)":    round(pf, 3),
        "平均盈利(元)":  round(float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0, 2),
        "平均亏损(元)":  round(float(abs(pnls[pnls < 0].mean())) if (pnls < 0).any() else 0, 2),
        "最大回撤(元)":  round(max_dd, 2),
        "平仓原因":      df["reason"].value_counts().to_dict(),
    }


# ─── 可视化 ──────────────────────────────────────────────────────────────────
def plot_results(bars: pd.DataFrame, result: dict) -> None:
    """绘制K线价格+三EMA、净值曲线、ADX图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("提示: 未安装 matplotlib，跳过图表绘制。")
        return

    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    trades_df = pd.DataFrame(result["trades"]) if result["trades"] else pd.DataFrame()
    equity    = result["equity"]
    idx       = range(len(bars))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 子图1：价格 + 三EMA + 入场标注
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
        f"ru2609 1分钟K线 EMA{FAST_EMA}/{SLOW_EMA}/{TREND_EMA} + ADX({ADX_PERIOD})>{ADX_THRESHOLD} 趋势策略"
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

    # 子图3：ADX 趋势强度图
    ax3 = axes[2]
    adx = bars["adx"].values
    ax3.plot(idx, adx,                    color="navy",  lw=0.8, label=f"ADX({ADX_PERIOD})")
    ax3.plot(idx, bars["di_plus"].values,  color="green", lw=0.6, linestyle="--", label="DI+")
    ax3.plot(idx, bars["di_minus"].values, color="red",   lw=0.6, linestyle="--", label="DI-")
    ax3.axhline(ADX_THRESHOLD, color="orange", lw=1.0, linestyle="-.",
                label=f"ADX阈值 {ADX_THRESHOLD}")
    ax3.axhline(25, color="gray", lw=0.8, linestyle=":",
                label="ADX=25(强趋势)")
    ax3.fill_between(idx, adx, ADX_THRESHOLD,
                     where=(np.nan_to_num(adx) >= ADX_THRESHOLD),
                     alpha=0.15, color="blue", label="有效趋势区域")
    ax3.set_title(f"ADX({ADX_PERIOD}) 趋势强度 — 仅ADX>{ADX_THRESHOLD}时允许入场")
    ax3.set_ylabel("ADX / DI 值")
    ax3.set_xlabel("1分钟K线序号")
    ax3.legend(fontsize=8, ncol=3)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / "backtest_result.png"
    plt.savefig(out_path, dpi=150)
    print(f"图表已保存至: {out_path}")
    plt.close()


# ─── 主程序 ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" ru2609 天然橡胶「EMA趋势 + ADX市场过滤」策略 回测")
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

    bars = compute_indicators(bars)
    n_long  = (bars["signal"] ==  1).sum()
    n_short = (bars["signal"] == -1).sum()
    n_adx_ok = (bars["signal"] != 0) & (bars["adx"].fillna(0) >= ADX_THRESHOLD)
    print(f"EMA穿越信号: 做多{n_long}次 / 做空{n_short}次")
    print(f"ADX>{ADX_THRESHOLD}有效信号: {n_adx_ok.sum()} 次")
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
                          "dir_str", "adx_at_entry", "pnl", "reason"]
                         ].round(1).to_string(index=False))
        print()
        out_csv = Path(__file__).parent / "trades.csv"
        trades_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"完整交易记录已保存至: {out_csv}")

    plot_results(bars, result)

    print()
    print("策略说明")
    print("─" * 40)
    net_tp = TAKE_PROFIT * CONTRACT_UNIT - COMMISSION * 2
    net_sl = STOP_LOSS   * CONTRACT_UNIT + COMMISSION * 2
    print(f"  策略类型    : 1分钟K线 EMA三线趋势 + ADX市场过滤")
    print(f"  EMA参数     : 快EMA{FAST_EMA} / 慢EMA{SLOW_EMA} / 趋势EMA{TREND_EMA}(1小时)")
    print(f"  信号条件    : EMA金叉/死叉 AND |差值|>={MIN_CROSS_DIFF}pt AND 收盘顺EMA{TREND_EMA}")
    print(f"  ADX过滤     : 信号K线ADX>{ADX_THRESHOLD}（趋势市才允许入场，屏蔽横盘震荡）")
    print(f"  止盈        : {TAKE_PROFIT}pt → 净盈利{net_tp}元/手")
    print(f"  止损        : {STOP_LOSS}pt  → 净亏损{net_sl}元/手")
    print(f"  盈亏比      : {net_tp/net_sl:.2f}:1（胜率>={net_sl/(net_tp+net_sl):.1%}即正期望）")
    print(f"  单边手续费  : {COMMISSION}元/手，合约乘数{CONTRACT_UNIT}吨/手")


if __name__ == "__main__":
    main()
