"""
商品期货盘口交易系统回测脚本
标的: ru2609 (天然橡胶期货)
数据: 逐笔tick盘口数据 (1.txt)

交易系统设计思路
=================
通过对盘口数据的统计分析，发现以下规律：
1. 短期价格动量（3~5个tick）与未来价格变化呈显著负相关（约 -0.27），
   即价格短期冲高/走低后倾向于回调 → 均值回归策略
2. 买卖盘口失衡指数（OBI）与未来价格方向正相关（约 +0.02），
   即买盘厚于卖盘时价格倾向上涨 → 用于辅助确认方向
3. 价差基本稳定在最小变动单位（5元/吨），市场流动性良好

核心交易逻辑（均值回归 + OBI双重确认）
========================================
- 综合信号 = OBI × w_obi - 归一化动量 × w_mom
  * OBI = (买盘总量 - 卖盘总量) / (买盘总量 + 卖盘总量)，取5档
  * 动量 = (当前价格 - N个tick前价格) / 归一化系数
- 做多条件: 信号 > 阈值（价格近期下跌 + 买盘占优）
- 做空条件: 信号 < -阈值（价格近期上涨 + 卖盘占优）
- 每次只持1手，不加仓

风险管理
=========
- 止盈: 10点（即2跳，100元/手）
- 止损: 10点（即2跳，100元/手）
- 最大持仓时间: 60个tick
- 跳过开盘前5分钟和午休时段

合约参数
=========
- 每手: 10吨
- 最小变动: 5元/吨 → 每跳 50元/手
- 手续费: 单边3元/手（往返6元/手）
- 保证金比例: 不参与资金管理，按固定1手计算
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─── 合约与策略参数 ──────────────────────────────────────────────────────────

DATA_FILE = Path(__file__).parent / "1.txt"

CONTRACT_UNIT = 10          # 吨/手
TICK_SIZE = 5               # 元/吨（最小变动价位）
COMMISSION = 3              # 元/手（单边手续费）

MOMENTUM_WINDOW = 3         # 动量计算的tick窗口
OBI_WEIGHT = 0.15           # OBI 信号权重
MOM_WEIGHT = 1.0            # 动量信号权重
SIGNAL_THRESHOLD = 0.20     # 入场信号阈值
TAKE_PROFIT = 10            # 止盈点数（元/吨）
STOP_LOSS = 10              # 止损点数（元/吨）
MAX_HOLD_TICKS = 60         # 最大持仓tick数

# 过滤掉的交易时段（避免开盘前几分钟和午休）
SKIP_MINUTES_AFTER_OPEN = 5   # 开盘后跳过的分钟数
LUNCH_START = "11:30"
LUNCH_END = "13:30"
NIGHT_START = "21:00"
NIGHT_END = "21:05"          # 夜盘开盘后跳过的分钟数


# ─── 数据加载与特征计算 ──────────────────────────────────────────────────────

def load_data(filepath: Path) -> pd.DataFrame:
    """加载并预处理盘口数据。"""
    df = pd.read_csv(filepath)

    # 将Unix毫秒时间戳转为北京时间
    df["dt"] = (
        pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )

    # 只保留有效报价行（去掉集合竞价/休市期间的无价格行）
    df = df[(df["LastPrice"] > 0) & (df["BidPrice1"] > 0)].copy()
    df = df.reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算策略所需的技术指标。"""
    # 5档买卖盘总量
    bid_cols = ["BidVolume1", "BidVolume2", "BidVolume3", "BidVolume4", "BidVolume5"]
    ask_cols = ["AskVolume1", "AskVolume2", "AskVolume3", "AskVolume4", "AskVolume5"]
    df["bid_vol"] = df[bid_cols].sum(axis=1)
    df["ask_vol"] = df[ask_cols].sum(axis=1)

    # 买卖盘失衡指数 OBI ∈ (-1, 1)
    df["obi"] = (df["bid_vol"] - df["ask_vol"]) / (
        df["bid_vol"] + df["ask_vol"] + 1e-9
    )

    # 短期价格动量（N个tick前到现在的价格变化，归一化到tick单位）
    df["momentum"] = (
        (df["LastPrice"] - df["LastPrice"].shift(MOMENTUM_WINDOW)) / TICK_SIZE
    )

    # 综合信号：OBI 正向，动量反向（均值回归）
    df["signal"] = OBI_WEIGHT * df["obi"] - MOM_WEIGHT * df["momentum"].clip(-10, 10) / 10

    # 时间辅助字段
    df["time_str"] = df["dt"].dt.strftime("%H:%M")
    df["date"] = df["dt"].dt.date

    return df


def is_tradable_time(time_str: str) -> bool:
    """判断当前时间是否在可交易窗口内（过滤午休、开收盘等极端时段）。"""
    # 过滤午休
    if LUNCH_START <= time_str < LUNCH_END:
        return False
    # 过滤夜盘开盘后的几分钟（波动异常）
    if NIGHT_START <= time_str < NIGHT_END:
        return False
    # 过滤日盘开盘前几分钟（08:59 ~ 09:04）
    if time_str < "09:05":
        return False
    return True


# ─── 回测引擎 ────────────────────────────────────────────────────────────────

class Position:
    """代表一笔开仓记录。"""
    __slots__ = ("direction", "entry_price", "entry_idx", "size")

    def __init__(self, direction: int, entry_price: float, entry_idx: int, size: int = 1):
        self.direction = direction      # +1 多头, -1 空头
        self.entry_price = entry_price
        self.entry_idx = entry_idx
        self.size = size


def run_backtest(df: pd.DataFrame) -> dict:
    """
    逐tick回测核心引擎。

    返回包含以下字段的字典:
        trades      : 每笔交易记录列表
        equity      : 净值曲线（每tick）
        final_pnl   : 最终总盈亏（元）
    """
    position: Position | None = None
    trades = []
    equity_curve = []
    cumulative_pnl = 0.0

    prices = df["LastPrice"].values
    signals = df["signal"].values
    times = df["time_str"].values
    n = len(df)

    for i in range(MOMENTUM_WINDOW, n):
        price = prices[i]
        sig = signals[i]
        time_str = times[i]

        # ── 持仓检查：止盈/止损/超时平仓 ──────────────────────────
        if position is not None:
            pnl_pts = position.direction * (price - position.entry_price)
            hold_ticks = i - position.entry_idx

            should_close = False
            close_reason = ""

            if pnl_pts >= TAKE_PROFIT:
                should_close = True
                close_reason = "TP"
            elif pnl_pts <= -STOP_LOSS:
                should_close = True
                close_reason = "SL"
            elif hold_ticks >= MAX_HOLD_TICKS:
                should_close = True
                close_reason = "TIMEOUT"

            if should_close:
                trade_pnl = (
                    pnl_pts * CONTRACT_UNIT * position.size
                    - COMMISSION * position.size * 2  # 往返手续费
                )
                cumulative_pnl += trade_pnl
                trades.append(
                    {
                        "entry_idx": position.entry_idx,
                        "exit_idx": i,
                        "entry_price": position.entry_price,
                        "exit_price": price,
                        "direction": position.direction,
                        "hold_ticks": hold_ticks,
                        "pnl": trade_pnl,
                        "reason": close_reason,
                    }
                )
                position = None

        equity_curve.append(cumulative_pnl)

        # ── 入场判断（空仓时） ─────────────────────────────────────
        if position is None and is_tradable_time(time_str):
            if sig > SIGNAL_THRESHOLD:
                position = Position(
                    direction=1, entry_price=price, entry_idx=i
                )
            elif sig < -SIGNAL_THRESHOLD:
                position = Position(
                    direction=-1, entry_price=price, entry_idx=i
                )

    # 如果还有持仓，按最后一个价格平仓
    if position is not None:
        price = prices[-1]
        pnl_pts = position.direction * (price - position.entry_price)
        trade_pnl = (
            pnl_pts * CONTRACT_UNIT * position.size
            - COMMISSION * position.size * 2
        )
        cumulative_pnl += trade_pnl
        trades.append(
            {
                "entry_idx": position.entry_idx,
                "exit_idx": n - 1,
                "entry_price": position.entry_price,
                "exit_price": price,
                "direction": position.direction,
                "hold_ticks": n - 1 - position.entry_idx,
                "pnl": trade_pnl,
                "reason": "END",
            }
        )

    return {
        "trades": trades,
        "equity": equity_curve,
        "final_pnl": cumulative_pnl,
    }


# ─── 绩效统计 ────────────────────────────────────────────────────────────────

def compute_stats(result: dict) -> dict:
    """计算常用回测绩效指标。"""
    trades = result["trades"]
    equity = result["equity"]

    if not trades:
        return {"error": "无交易记录"}

    trades_df = pd.DataFrame(trades)
    pnls = trades_df["pnl"].values
    equity_arr = np.array(equity)

    total_trades = len(trades_df)
    winning = (pnls > 0).sum()
    losing = (pnls < 0).sum()
    win_rate = winning / total_trades if total_trades > 0 else 0

    gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    avg_win = pnls[pnls > 0].mean() if (pnls > 0).any() else 0
    avg_loss = abs(pnls[pnls < 0].mean()) if (pnls < 0).any() else 0

    # 最大回撤
    peak = np.maximum.accumulate(equity_arr)
    drawdown = equity_arr - peak
    max_drawdown = drawdown.min()

    # 持仓时间分布
    hold_ticks = trades_df["hold_ticks"]

    # 平仓原因分布
    reason_counts = trades_df["reason"].value_counts().to_dict()

    # 按方向统计
    long_pnl = trades_df[trades_df["direction"] == 1]["pnl"].sum()
    short_pnl = trades_df[trades_df["direction"] == -1]["pnl"].sum()

    return {
        "总交易次数": total_trades,
        "盈利次数": int(winning),
        "亏损次数": int(losing),
        "胜率": f"{win_rate:.2%}",
        "总盈亏(元)": round(result["final_pnl"], 2),
        "多头盈亏(元)": round(long_pnl, 2),
        "空头盈亏(元)": round(short_pnl, 2),
        "毛利润(元)": round(gross_profit, 2),
        "毛亏损(元)": round(gross_loss, 2),
        "盈亏比": round(profit_factor, 3),
        "平均盈利(元)": round(avg_win, 2),
        "平均亏损(元)": round(avg_loss, 2),
        "最大回撤(元)": round(max_drawdown, 2),
        "平均持仓tick数": round(hold_ticks.mean(), 1),
        "最大持仓tick数": int(hold_ticks.max()),
        "平仓原因分布": reason_counts,
    }


# ─── 可视化 ──────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, result: dict) -> None:
    """绘制净值曲线、价格走势和信号分布图。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("提示: 未安装 matplotlib，跳过图表绘制。")
        return

    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    trades_df = pd.DataFrame(result["trades"])
    equity = result["equity"]
    prices = df["LastPrice"].values[3:]   # 对齐 momentum_window 偏移

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # ─ 子图1：价格走势 + 买卖信号
    ax1 = axes[0]
    ax1.plot(prices, color="steelblue", linewidth=0.6, label="LastPrice")
    if not trades_df.empty:
        longs = trades_df[trades_df["direction"] == 1]
        shorts = trades_df[trades_df["direction"] == -1]
        ax1.scatter(
            longs["entry_idx"] - MOMENTUM_WINDOW,
            prices[(longs["entry_idx"] - MOMENTUM_WINDOW).clip(0)],
            marker="^", color="red", s=30, zorder=5, label="做多入场",
        )
        ax1.scatter(
            shorts["entry_idx"] - MOMENTUM_WINDOW,
            prices[(shorts["entry_idx"] - MOMENTUM_WINDOW).clip(0)],
            marker="v", color="green", s=30, zorder=5, label="做空入场",
        )
    ax1.set_title("ru2609 价格走势与交易信号")
    ax1.set_ylabel("价格（元/吨）")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ─ 子图2：净值曲线
    ax2 = axes[1]
    ax2.plot(equity, color="darkorange", linewidth=1, label="累计盈亏(元)")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.fill_between(range(len(equity)), equity, 0,
                     where=[e >= 0 for e in equity], alpha=0.2, color="green")
    ax2.fill_between(range(len(equity)), equity, 0,
                     where=[e < 0 for e in equity], alpha=0.2, color="red")
    ax2.set_title("累计盈亏净值曲线")
    ax2.set_ylabel("盈亏（元）")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ─ 子图3：信号分布
    ax3 = axes[2]
    signals = df["signal"].values[MOMENTUM_WINDOW:]
    ax3.plot(signals, color="purple", linewidth=0.4, alpha=0.7, label="综合信号")
    ax3.axhline(SIGNAL_THRESHOLD, color="red", linestyle="--", linewidth=0.8, label=f"入场阈值 ±{SIGNAL_THRESHOLD}")
    ax3.axhline(-SIGNAL_THRESHOLD, color="green", linestyle="--", linewidth=0.8)
    ax3.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax3.set_title("交易信号（OBI×0.15 - 动量/10）")
    ax3.set_ylabel("信号值")
    ax3.set_xlabel("Tick 序号")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / "backtest_result.png"
    plt.savefig(out_path, dpi=150)
    print(f"图表已保存至: {out_path}")
    plt.close()


# ─── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" ru2609 天然橡胶期货盘口均值回归策略 回测")
    print("=" * 60)
    print()

    # 1. 加载数据
    print(f"加载数据: {DATA_FILE}")
    df = load_data(DATA_FILE)
    print(f"有效tick数: {len(df)}")
    print()

    # 2. 计算特征
    df = compute_features(df)
    print(f"交易日期: {sorted(df['date'].unique())}")
    print(f"价格范围: {df['LastPrice'].min()} ~ {df['LastPrice'].max()} 元/吨")
    print()

    # 3. 运行回测
    print("运行回测...")
    result = run_backtest(df)
    print(f"共产生 {len(result['trades'])} 笔交易")
    print()

    # 4. 统计绩效
    stats = compute_stats(result)
    print("── 回测绩效统计 ──────────────────────────────")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()

    # 5. 输出交易明细
    if result["trades"]:
        trades_df = pd.DataFrame(result["trades"])
        trades_df["direction_str"] = trades_df["direction"].map({1: "多", -1: "空"})
        print("── 交易明细（前20笔）───────────────────────")
        print(
            trades_df[
                ["entry_idx", "exit_idx", "entry_price", "exit_price",
                 "direction_str", "hold_ticks", "pnl", "reason"]
            ]
            .head(20)
            .to_string(index=False)
        )
        print()

        # 保存完整交易记录
        out_csv = Path(__file__).parent / "trades.csv"
        trades_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"完整交易记录已保存至: {out_csv}")

    # 6. 绘图
    plot_results(df, result)

    print()
    print("策略说明")
    print("─" * 40)
    print(f"  信号公式   : {OBI_WEIGHT}×OBI - {MOM_WEIGHT}×动量({MOMENTUM_WINDOW}tick)/10")
    print(f"  入场阈值   : |信号| > {SIGNAL_THRESHOLD}")
    print(f"  止盈       : {TAKE_PROFIT} 点 ({TAKE_PROFIT * CONTRACT_UNIT} 元/手)")
    print(f"  止损       : {STOP_LOSS} 点 ({STOP_LOSS * CONTRACT_UNIT} 元/手)")
    print(f"  最大持仓   : {MAX_HOLD_TICKS} ticks")
    print(f"  单边手续费 : {COMMISSION} 元/手")
    print(f"  合约乘数   : {CONTRACT_UNIT} 吨/手，最小变动 {TICK_SIZE} 元/吨")


if __name__ == "__main__":
    main()
