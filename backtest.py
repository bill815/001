#!/usr/bin/env python3
"""
橡胶期货（ru2609）OBIT 回测系统
盘口失衡驱动的趋势跟踪系统（Order Book Imbalance-based Trend, OBIT）

依据 1.txt 逐笔数据，按以下规则回测：
  - 模块1：OBI（盘口失衡指标）
  - 模块2：大单墙识别
  - 模块3：趋势过滤器（价格动量 + 持仓量方向）
  - 模块4：完整交易规则（做多/做空入场、止损、止盈、移动止损）
  - 模块5：风险控制框架
"""

import csv
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import os

# ─────────────────────────── 参数配置 ────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "1.txt")

INITIAL_CAPITAL = 1_000_000.0          # 初始资金（元）
CONTRACT_MULTIPLIER = 10               # 橡胶合约乘数（10元/点）
TICK_SIZE = 5                          # 最小变动价位（5元/吨）

OBI_LONG_THRESHOLD = 0.20             # OBI 做多阈值
OBI_SHORT_THRESHOLD = -0.20           # OBI 做空阈值
MOMENTUM_WINDOW = 20                  # 动量计算 tick 窗口
WALL_MULTIPLIER = 2.5                 # 大单墙判定倍数（均值的 N 倍）

STOP_LOSS_POINTS = 15                 # 止损点数（3 跳）
TP1_POINTS = 20                       # 第一止盈点数（4 跳）
TP2_POINTS = 40                       # 第二止盈点数（8 跳）
TRAILING_STOP_POINTS = 10            # 移动止损间距

MAX_RISK_PER_TRADE = 0.01            # 单笔最大亏损比例（1%）
MAX_DAILY_LOSS = 0.03                # 日最大亏损比例（3%）
MAX_POSITION_RATIO = 0.20            # 单笔最大仓位比例（20%）
MAX_DAILY_TRADES = 8                 # 日内最大交易次数
MARGIN_RATIO = 0.15                  # 橡胶合约保证金比例（约 15%）

MIN_TICKS_BEFORE_TRADE = MOMENTUM_WINDOW  # 至少积累 N 个 tick 后才允许开仓
# ─────────────────────────── 数据结构 ────────────────────────────

@dataclass
class Tick:
    trading_day: str
    last_price: float
    volume: int
    open_interest: float
    bid_prices: list   # [bid1, bid2, bid3, bid4, bid5]
    bid_vols: list     # [vol1, vol2, vol3, vol4, vol5]
    ask_prices: list   # [ask1, ask2, ask3, ask4, ask5]
    ask_vols: list     # [vol1, vol2, vol3, vol4, vol5]
    timestamp: int


@dataclass
class Position:
    direction: str          # 'long' or 'short'
    entry_price: float
    lots: int
    stop_loss: float
    tp1: float
    tp2: float
    tp1_hit: bool = False
    trailing_stop: Optional[float] = None
    entry_tick_idx: int = 0


@dataclass
class Trade:
    direction: str
    entry_price: float
    exit_price: float
    lots: int
    pnl: float
    entry_tick_idx: int
    exit_tick_idx: int
    exit_reason: str
    trading_day: str


# ─────────────────────────── 指标计算 ────────────────────────────

def calc_obi(bid_vols: list, ask_vols: list) -> float:
    """盘口失衡指标 OBI = (买方总量 - 卖方总量) / (买方总量 + 卖方总量)"""
    bid_total = sum(v for v in bid_vols if v > 0)
    ask_total = sum(v for v in ask_vols if v > 0)
    total = bid_total + ask_total
    if total == 0:
        return 0.0
    return (bid_total - ask_total) / total


def calc_momentum(price_window: deque) -> float:
    """价格动量 = (最新价 - N tick前价格) / N"""
    if len(price_window) < MOMENTUM_WINDOW:
        return 0.0
    prices = list(price_window)
    return (prices[-1] - prices[0]) / MOMENTUM_WINDOW


def calc_avg_depth(bid_vols: list, ask_vols: list) -> float:
    """计算五档均值挂单量（用于大单墙判定）"""
    all_vols = [v for v in bid_vols + ask_vols if v > 0]
    if not all_vols:
        return 0.0
    return sum(all_vols) / len(all_vols)


def detect_walls(bid_prices: list, bid_vols: list,
                 ask_prices: list, ask_vols: list,
                 avg_depth: float):
    """
    大单墙识别：挂单量 >= 均值 * WALL_MULTIPLIER 视为大单墙。
    返回 (bid_walls, ask_walls) 各为 (price, vol) 列表。
    """
    threshold = avg_depth * WALL_MULTIPLIER
    if threshold <= 0:
        return [], []
    bid_walls = [(p, v) for p, v in zip(bid_prices, bid_vols)
                 if v >= threshold and p > 0]
    ask_walls = [(p, v) for p, v in zip(ask_prices, ask_vols)
                 if v >= threshold and p > 0]
    return bid_walls, ask_walls


def near_wall(current_price: float, walls: list, direction: str,
              n_levels: int = 5) -> bool:
    """
    判断当前价格是否在大单墙的 n_levels 档以内。
    direction='ask': 当前价格在卖方墙正上方 n_levels*TICK_SIZE 内 → 做多受阻
    direction='bid': 当前价格在买方墙正下方 n_levels*TICK_SIZE 内 → 做空受阻
    """
    for wall_price, _ in walls:
        dist = abs(current_price - wall_price)
        if dist <= n_levels * TICK_SIZE:
            return True
    return False


# ─────────────────────────── 数据解析 ────────────────────────────

def parse_data(filepath: str) -> list:
    """解析 1.txt，返回 Tick 列表（跳过首行 header，跳过无效 tick）。"""
    ticks = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)   # 跳过 header

        # 建立列索引映射
        col = {name.strip(): idx for idx, name in enumerate(header)}

        for row in reader:
            if not row or len(row) < 38:
                continue
            try:
                last_price = float(row[col["LastPrice"]])
                if last_price <= 0:
                    continue
                volume = int(float(row[col["Volume"]]))
                oi = float(row[col["OpenInterest"]])
                ts = int(row[col["Timestamp"]])
                tday = row[col["TradingDay"]].strip()

                # 五档买卖价/量（列顺序：1,2,3,4,5）
                bp = [
                    float(row[col["BidPrice1"]]),
                    float(row[col["BidPrice2"]]),
                    float(row[col["BidPrice3"]]),
                    float(row[col["BidPrice4"]]),
                    float(row[col["BidPrice5"]]),
                ]
                bv = [
                    int(float(row[col["BidVolume1"]])),
                    int(float(row[col["BidVolume2"]])),
                    int(float(row[col["BidVolume3"]])),
                    int(float(row[col["BidVolume4"]])),
                    int(float(row[col["BidVolume5"]])),
                ]
                ap = [
                    float(row[col["AskPrice1"]]),
                    float(row[col["AskPrice2"]]),
                    float(row[col["AskPrice3"]]),
                    float(row[col["AskPrice4"]]),
                    float(row[col["AskPrice5"]]),
                ]
                av = [
                    int(float(row[col["AskVolume1"]])),
                    int(float(row[col["AskVolume2"]])),
                    int(float(row[col["AskVolume3"]])),
                    int(float(row[col["AskVolume4"]])),
                    int(float(row[col["AskVolume5"]])),
                ]

                ticks.append(Tick(
                    trading_day=tday,
                    last_price=last_price,
                    volume=volume,
                    open_interest=oi,
                    bid_prices=bp,
                    bid_vols=bv,
                    ask_prices=ap,
                    ask_vols=av,
                    timestamp=ts,
                ))
            except (ValueError, IndexError):
                continue
    return ticks


# ─────────────────────────── 仓位计算 ────────────────────────────

def calc_lots(nav: float, entry_price: float) -> int:
    """
    根据风险控制框架计算开仓手数：
    - 单笔最大亏损 = nav * MAX_RISK_PER_TRADE
    - 单笔止损额 = STOP_LOSS_POINTS * CONTRACT_MULTIPLIER
    - 手数 = floor(单笔最大亏损 / 单笔止损额)
    - 同时不超过 nav * MAX_POSITION_RATIO / (entry_price * CONTRACT_MULTIPLIER * MARGIN_RATIO)
    """
    max_loss_amount = nav * MAX_RISK_PER_TRADE
    loss_per_lot = STOP_LOSS_POINTS * CONTRACT_MULTIPLIER
    lots_by_risk = math.floor(max_loss_amount / loss_per_lot)

    margin_per_lot = entry_price * CONTRACT_MULTIPLIER * MARGIN_RATIO
    max_margin = nav * MAX_POSITION_RATIO
    lots_by_margin = math.floor(max_margin / margin_per_lot) if margin_per_lot > 0 else 0

    lots = min(lots_by_risk, lots_by_margin)
    return max(1, lots)  # 至少 1 手


# ─────────────────────────── 主回测引擎 ────────────────────────────

def run_backtest(ticks: list) -> dict:
    """
    主回测循环，返回回测结果字典。
    按交易日分组处理，每日重置日内计数器。
    """
    nav = INITIAL_CAPITAL
    position: Optional[Position] = None
    trades: list = []

    price_window: deque = deque(maxlen=MOMENTUM_WINDOW)
    oi_window: deque = deque(maxlen=MOMENTUM_WINDOW)

    # 滚动平均深度（用于大单墙判定）
    depth_window: deque = deque(maxlen=100)

    # 日内统计
    current_day = None
    daily_trades = 0
    daily_pnl = 0.0
    day_start_nav = nav
    daily_stopped = False
    tick_idx_in_day = 0

    total_ticks = len(ticks)

    for idx, tick in enumerate(ticks):
        # ── 日切换重置 ──
        if tick.trading_day != current_day:
            current_day = tick.trading_day
            daily_trades = 0
            daily_pnl = 0.0
            day_start_nav = nav
            daily_stopped = False
            tick_idx_in_day = 0
            price_window.clear()
            oi_window.clear()
            depth_window.clear()

        tick_idx_in_day += 1

        # 更新滚动窗口前先快照（用于信号判断）
        prev_prices = list(price_window)
        prev_oi = list(oi_window)

        # 更新滚动窗口
        price_window.append(tick.last_price)
        oi_window.append(tick.open_interest)

        avg_depth = calc_avg_depth(tick.bid_vols, tick.ask_vols)
        if avg_depth > 0:
            depth_window.append(avg_depth)
        rolling_avg_depth = sum(depth_window) / len(depth_window) if depth_window else 1.0

        # ── 检查已有持仓的出场条件 ──
        if position is not None:
            pos = position
            price = tick.last_price

            exit_price = None
            exit_reason = None
            exit_lots = 0

            if pos.direction == "long":
                # 止损
                if price <= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = "stop_loss"
                    exit_lots = pos.lots
                # 第一止盈（平半仓）
                elif not pos.tp1_hit and price >= pos.tp1:
                    half = max(1, pos.lots // 2)
                    pnl = (pos.tp1 - pos.entry_price) * CONTRACT_MULTIPLIER * half
                    trades.append(Trade(
                        direction=pos.direction,
                        entry_price=pos.entry_price,
                        exit_price=pos.tp1,
                        lots=half,
                        pnl=pnl,
                        entry_tick_idx=pos.entry_tick_idx,
                        exit_tick_idx=idx,
                        exit_reason="tp1",
                        trading_day=tick.trading_day,
                    ))
                    nav += pnl
                    daily_pnl += pnl
                    pos.lots -= half
                    pos.tp1_hit = True
                    pos.stop_loss = pos.entry_price  # 移至成本价
                    if pos.lots <= 0:
                        position = None
                        continue
                # 第二止盈（切换移动止损模式，不立即平仓）
                elif pos.tp1_hit and pos.trailing_stop is None and price >= pos.tp2:
                    pos.trailing_stop = price - TRAILING_STOP_POINTS
                # 移动止损跟踪
                elif pos.tp1_hit and pos.trailing_stop is not None:
                    new_trail = price - TRAILING_STOP_POINTS
                    if new_trail > pos.trailing_stop:
                        pos.trailing_stop = new_trail
                    if price <= pos.trailing_stop:
                        exit_price = pos.trailing_stop
                        exit_reason = "trailing_stop"
                        exit_lots = pos.lots

            elif pos.direction == "short":
                # 止损
                if price >= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = "stop_loss"
                    exit_lots = pos.lots
                # 第一止盈（平半仓）
                elif not pos.tp1_hit and price <= pos.tp1:
                    half = max(1, pos.lots // 2)
                    pnl = (pos.entry_price - pos.tp1) * CONTRACT_MULTIPLIER * half
                    trades.append(Trade(
                        direction=pos.direction,
                        entry_price=pos.entry_price,
                        exit_price=pos.tp1,
                        lots=half,
                        pnl=pnl,
                        entry_tick_idx=pos.entry_tick_idx,
                        exit_tick_idx=idx,
                        exit_reason="tp1",
                        trading_day=tick.trading_day,
                    ))
                    nav += pnl
                    daily_pnl += pnl
                    pos.lots -= half
                    pos.tp1_hit = True
                    pos.stop_loss = pos.entry_price  # 移至成本价
                    if pos.lots <= 0:
                        position = None
                        continue
                # 第二止盈（切换移动止损模式，不立即平仓）
                elif pos.tp1_hit and pos.trailing_stop is None and price <= pos.tp2:
                    pos.trailing_stop = price + TRAILING_STOP_POINTS
                # 移动止损跟踪
                elif pos.tp1_hit and pos.trailing_stop is not None:
                    new_trail = price + TRAILING_STOP_POINTS
                    if new_trail < pos.trailing_stop:
                        pos.trailing_stop = new_trail
                    if price >= pos.trailing_stop:
                        exit_price = pos.trailing_stop
                        exit_reason = "trailing_stop"
                        exit_lots = pos.lots

            # 执行出场
            if exit_price is not None and exit_lots > 0:
                if pos.direction == "long":
                    pnl = (exit_price - pos.entry_price) * CONTRACT_MULTIPLIER * exit_lots
                else:
                    pnl = (pos.entry_price - exit_price) * CONTRACT_MULTIPLIER * exit_lots
                trades.append(Trade(
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    lots=exit_lots,
                    pnl=pnl,
                    entry_tick_idx=pos.entry_tick_idx,
                    exit_tick_idx=idx,
                    exit_reason=exit_reason,
                    trading_day=tick.trading_day,
                ))
                nav += pnl
                daily_pnl += pnl
                position = None

        # ── 日止损检查 ──
        if daily_stopped:
            continue
        if daily_pnl < -day_start_nav * MAX_DAILY_LOSS:
            daily_stopped = True
            continue

        # ── 已有持仓则不开新仓 ──
        if position is not None:
            continue

        # ── 交易次数限制 ──
        if daily_trades >= MAX_DAILY_TRADES:
            continue

        # ── 积累足够 tick 后才可交易 ──
        if tick_idx_in_day <= MIN_TICKS_BEFORE_TRADE:
            continue

        # ── 计算信号指标 ──
        obi = calc_obi(tick.bid_vols, tick.ask_vols)
        momentum = calc_momentum(price_window)

        # 使用更新前的快照判断趋势方向
        oi_increased = (prev_oi[-1] > prev_oi[0]) if len(prev_oi) == MOMENTUM_WINDOW else False

        # 近 20 tick 最高/最低价（不含当前 tick，用于突破判断）
        if len(prev_prices) < MOMENTUM_WINDOW:
            continue
        high_20 = max(prev_prices)
        low_20 = min(prev_prices)

        bid_walls, ask_walls = detect_walls(
            tick.bid_prices, tick.bid_vols,
            tick.ask_prices, tick.ask_vols,
            rolling_avg_depth,
        )

        price = tick.last_price

        # ── 做多信号 ──
        long_signal = (
            obi > OBI_LONG_THRESHOLD                     # ① OBI 买方占优
            and price > high_20                          # ② 突破近 20 tick 最高价
            and oi_increased                             # ③ 持仓量增加
            and not near_wall(price, ask_walls, "ask")   # ④ 不在卖方大单墙正上方
            and momentum > 0                             # 动量过滤
        )

        # ── 做空信号 ──
        short_signal = (
            obi < OBI_SHORT_THRESHOLD                   # ① OBI 卖方占优
            and price < low_20                          # ② 跌破近 20 tick 最低价
            and oi_increased                            # ③ 持仓量增加
            and not near_wall(price, bid_walls, "bid")  # ④ 不在买方大单墙正下方
            and momentum < 0                            # 动量过滤
        )

        if long_signal or short_signal:
            direction = "long" if long_signal else "short"
            lots = calc_lots(nav, price)

            if direction == "long":
                sl = price - STOP_LOSS_POINTS
                tp1 = price + TP1_POINTS
                tp2 = price + TP2_POINTS
            else:
                sl = price + STOP_LOSS_POINTS
                tp1 = price - TP1_POINTS
                tp2 = price - TP2_POINTS

            position = Position(
                direction=direction,
                entry_price=price,
                lots=lots,
                stop_loss=sl,
                tp1=tp1,
                tp2=tp2,
                entry_tick_idx=idx,
            )
            daily_trades += 1

    # ── 收盘强制平仓（每日末若有持仓）──
    if position is not None:
        last_tick = ticks[-1]
        price = last_tick.last_price
        pos = position
        if pos.direction == "long":
            pnl = (price - pos.entry_price) * CONTRACT_MULTIPLIER * pos.lots
        else:
            pnl = (pos.entry_price - price) * CONTRACT_MULTIPLIER * pos.lots
        trades.append(Trade(
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=price,
            lots=pos.lots,
            pnl=pnl,
            entry_tick_idx=pos.entry_tick_idx,
            exit_tick_idx=total_ticks - 1,
            exit_reason="eod_close",
            trading_day=last_tick.trading_day,
        ))
        nav += pnl
        position = None

    return {
        "trades": trades,
        "final_nav": nav,
        "initial_capital": INITIAL_CAPITAL,
    }


# ─────────────────────────── 报告输出 ────────────────────────────

def print_report(result: dict) -> None:
    trades = result["trades"]
    initial_capital = result["initial_capital"]
    final_nav = result["final_nav"]

    separator = "=" * 70

    print(separator)
    print("  橡胶期货（ru2609）OBIT 回测报告")
    print(separator)

    # ── 逐笔交易明细 ──
    print("\n【逐笔交易明细】")
    print(f"{'序号':>4}  {'日期':>10}  {'方向':>4}  {'入场价':>8}  {'出场价':>8}  "
          f"{'手数':>4}  {'盈亏(元)':>10}  {'出场原因'}")
    print("-" * 75)

    total_pnl = 0.0
    wins = 0
    losses = 0
    daily_stats: dict = {}

    for i, t in enumerate(trades, 1):
        dir_label = "做多" if t.direction == "long" else "做空"
        reason_map = {
            "stop_loss": "止损",
            "tp1": "第一止盈",
            "tp2": "第二止盈",
            "trailing_stop": "移动止损",
            "eod_close": "收盘平仓",
        }
        reason = reason_map.get(t.exit_reason, t.exit_reason)
        total_pnl += t.pnl
        if t.pnl > 0:
            wins += 1
        elif t.pnl < 0:
            losses += 1

        # 日统计
        day = t.trading_day
        if day not in daily_stats:
            daily_stats[day] = {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0}
        daily_stats[day]["trades"] += 1
        daily_stats[day]["pnl"] += t.pnl
        if t.pnl > 0:
            daily_stats[day]["wins"] += 1
        elif t.pnl < 0:
            daily_stats[day]["losses"] += 1

        pnl_str = f"{t.pnl:>+10.0f}"
        print(f"{i:>4}  {day:>10}  {dir_label:>4}  {t.entry_price:>8.0f}  "
              f"{t.exit_price:>8.0f}  {t.lots:>4}  {pnl_str}  {reason}")

    total_trades = len(trades)
    breakevens = total_trades - wins - losses

    print(separator)

    # ── 分日汇总 ──
    print("\n【分日汇总】")
    print(f"{'日期':>10}  {'交易次数':>8}  {'盈利次数':>8}  {'亏损次数':>8}  {'日盈亏(元)':>12}")
    print("-" * 55)
    for day, ds in sorted(daily_stats.items()):
        print(f"{day:>10}  {ds['trades']:>8}  {ds['wins']:>8}  {ds['losses']:>8}  "
              f"{ds['pnl']:>+12.0f}")

    print(separator)

    # ── 整体绩效 ──
    net_return = (final_nav - initial_capital) / initial_capital * 100
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0.0
    avg_win = sum(t.pnl for t in trades if t.pnl > 0) / wins if wins > 0 else 0
    avg_loss = sum(t.pnl for t in trades if t.pnl < 0) / losses if losses > 0 else 0
    profit_factor = (
        sum(t.pnl for t in trades if t.pnl > 0) /
        abs(sum(t.pnl for t in trades if t.pnl < 0))
        if losses > 0 and sum(t.pnl for t in trades if t.pnl < 0) != 0 else float("inf")
    )
    avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # 最大连续亏损
    max_consecutive_loss = 0
    cur_consecutive_loss = 0
    for t in trades:
        if t.pnl < 0:
            cur_consecutive_loss += 1
            max_consecutive_loss = max(max_consecutive_loss, cur_consecutive_loss)
        else:
            cur_consecutive_loss = 0

    # 最大回撤（简单计算净值序列）
    nav_series = [initial_capital]
    running_nav = initial_capital
    for t in trades:
        running_nav += t.pnl
        nav_series.append(running_nav)
    peak = nav_series[0]
    max_drawdown = 0.0
    for v in nav_series:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > max_drawdown:
            max_drawdown = dd

    print("\n【整体绩效汇总】")
    print(f"  初始资金          : {initial_capital:>12,.0f} 元")
    print(f"  最终净值          : {final_nav:>12,.0f} 元")
    print(f"  总盈亏            : {total_pnl:>+12,.0f} 元")
    print(f"  净收益率          : {net_return:>+12.2f} %")
    print(f"  总交易次数        : {total_trades:>12}")
    print(f"  盈利次数          : {wins:>12}")
    print(f"  亏损次数          : {losses:>12}")
    print(f"  平局次数          : {breakevens:>12}")
    print(f"  胜率              : {win_rate:>12.1f} %")
    print(f"  平均盈利          : {avg_win:>+12.0f} 元")
    print(f"  平均亏损          : {avg_loss:>+12.0f} 元")
    print(f"  盈亏比            : {avg_rr:>12.2f}")
    print(f"  盈利因子          : {profit_factor:>12.2f}")
    print(f"  最大连续亏损次数  : {max_consecutive_loss:>12}")
    print(f"  最大回撤          : {max_drawdown:>12.2f} %")
    print(separator)

    # ── 系统参数回顾 ──
    print("\n【系统参数】")
    print(f"  OBI 做多阈值      : > {OBI_LONG_THRESHOLD}")
    print(f"  OBI 做空阈值      : < {OBI_SHORT_THRESHOLD}")
    print(f"  动量计算窗口      : {MOMENTUM_WINDOW} ticks")
    print(f"  大单墙判定倍数    : {WALL_MULTIPLIER}×均值")
    print(f"  止损点数          : {STOP_LOSS_POINTS} 点")
    print(f"  第一止盈          : {TP1_POINTS} 点")
    print(f"  第二止盈          : {TP2_POINTS} 点")
    print(f"  移动止损间距      : {TRAILING_STOP_POINTS} 点")
    print(f"  单笔最大风险      : {MAX_RISK_PER_TRADE*100:.0f}% 净值")
    print(f"  日止损上限        : {MAX_DAILY_LOSS*100:.0f}% 净值")
    print(f"  日内最大交易次数  : {MAX_DAILY_TRADES}")
    print(separator)


# ─────────────────────────── 入口 ────────────────────────────────

if __name__ == "__main__":
    print(f"正在加载数据：{DATA_FILE}")
    ticks = parse_data(DATA_FILE)
    print(f"共加载 {len(ticks)} 个有效 tick")

    days = sorted(set(t.trading_day for t in ticks))
    print(f"交易日：{', '.join(days)}\n")

    result = run_backtest(ticks)
    print_report(result)
