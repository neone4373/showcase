import json
import os
from logging import Logger

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from src.utils.sql_helpers import get_strategies
from src.zkbot.liquidity import Liquidity
from src.zkbot.pingpong_strategies.deep_dynamic_ping_pong_strategy import Order, Orders

pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 500)


class DeepCandles:
    """
    Class for storing recent candles for future analysis
    """
    def __init__(self, model_name="lstm", max_interval=250, last_row=None):
        self.recent_candles_df = pd.DataFrame()
        self.base_df = pd.DataFrame()
        self.max_interval = max_interval
        self.model_name = model_name
        self.last_row = {"pm": 0.01, "sl": -0.01} if last_row is None else last_row

    @property
    def latest_profit_and_loss(self):
        return [self.last_row.get(k, None) for k in ("pm", "sl")]

    @property
    def df(self):
        return self.recent_candles_df


class Market:
    """
    Object to represent the market data object
    """
    def __init__(self, market_info=None):
        self.alias = None
        self.base_fee = np.nan
        self.quote_fee = np.nan
        self.base_asset_id = None
        self.quote_asset_id = None
        self.base_decimals = 0
        self.base_symbol = ""
        self.quote_decimals = 0
        self.quote_symbol = ""
        if market_info is not None:
            self.update_market(market_info)

    @property
    def base_quote(self):
        return [self.base_symbol, self.quote_symbol]

    def update_market(self, market_info):
        self.alias, self.base_fee, self.quote_fee, base_asset, quote_asset, \
            self.base_asset_id, self.quote_asset_id = \
            [market_info[k] for k in ["alias", "baseFee", "quoteFee", "baseAsset",
                                      "quoteAsset", "baseAssetId", "quoteAssetId"]]
        self.base_decimals, self.base_symbol = [base_asset[k] for k in ["decimals", "symbol"]]
        self.quote_decimals, self.quote_symbol = [quote_asset[k] for k in ["decimals", "symbol"]]


class Fill:
    def __init__(self, raw_fill_info=None):
        self._fills_schema = ["chain_id", "id", "market", "side", "price", "base_quantity", "fill_status", "tx_hash",
                              "taker_user_id", "maker_user_id", "fee_amount", "fee_token"]
        self._raw_fill_info = raw_fill_info
        self.fill = self.append_fills(raw_fill_info) if raw_fill_info is not None else {}

    def append_fills(self, raw_fill):
        if type(raw_fill) == list:
            fill_ = {a: b for a, b in zip(self._fills_schema, raw_fill)}
        else:
            fill_ = raw_fill
        fill = fill_.copy()
        price, fee_amount, base_quantity, quote_quantity = \
            [fill.get(k, np.nan) for k in ('price', "fee_amount", 'base_quantity', "quote_quantity")]
        side, fee_token, fill_status = \
            [fill.get(k, "") for k in ('side', "fee_token", "fill_status")]
        fill_status = "f" if fill_status == "" else fill_status
        if fill_status == "f":
            if side == 's':
                base_fee = fee_amount if fee_amount is not None else np.nan
                quote_quantity = (base_quantity - base_fee) * price
            else:
                if np.isnan(quote_quantity):
                    base_fee = fee_amount / price if fee_amount is not None else np.nan
                    quote_quantity = (base_quantity + base_fee) * price
                else:
                    base_fee = fee_amount if fee_amount is not None else np.nan
                    base_quantity = (quote_quantity - base_fee) / price

            real_price = quote_quantity / base_quantity
            fill["quote_quantity"] = quote_quantity
            fill["base_fee"] = base_fee
            fill["real_price"] = real_price
        return fill

    def get_fee(self, target, default_fee):
        fee = default_fee
        potential_fee = self.fill.get("fee_amount", np.nan)
        if type(potential_fee) == float and not np.isnan(potential_fee) and self.fill.get("fee_token", "") == target:
            fee = potential_fee
        return fee

    @property
    def id(self):
        return self.fill.get("id", 0)

    @property
    def side(self):
        return self.fill.get("side", "")

    @property
    def price(self):
        return self.fill.get("price")

    @property
    def fee_amount(self):
        return self.fill.get("fee_amount")

    @property
    def fee_token(self):
        return self.fill.get("fee_token")

    @property
    def status(self):
        return self.fill.get("fill_status")

    @property
    def price_base_quote_quantity(self):
        return [self.fill.get(k) for k in ["price", "base_quantity", "quote_quantity"]]

    def update_price(self, new_price, new_target, new_amount):
        fill_info = self.fill
        fill_info["price"] = new_price
        fill_info[new_target] = new_amount
        self.fill = self.append_fills(fill_info)


class Fills:
    def __init__(self):
        self.fills = []
        self.matched_fill = Fill()

    def add_fill(self, fill: Fill):
        if fill.fill not in [fill_.fill for fill_ in self.fills]:
            if fill.status == "m":
                self.matched_fill = fill
            else:
                self.fills.append(fill)
        return fill not in self.fills

    def finalize_matched_fill(self, fill_status):
        new_fill = self.matched_fill.fill
        new_fill["fill_status"] = "f"
        new_fill["tx_hash"] = fill_status[3]
        new_fill["price"] = float(new_fill["price"])
        new_fill["fee_amount"] = fill_status[5]
        new_fill["fee_token"] = fill_status[6]
        self.add_fill(Fill(new_fill))
        self.matched_fill = Fill()

    @property
    def last(self) -> Fill:
        if len(self.fills) > 0:
            return self.fills[-1]
        else:
            return Fill()


class PositionOrder(Order):
    def __init__(self, order_info: dict = None, simulation=False, expiration_buffer=4, order_fudge=0.002,
                 slippage=0.002, quote_quality_variance=0.0010):
        super().__init__(order_info, simulation=simulation, expiration_buffer=expiration_buffer,
                         order_fudge=order_fudge, slippage=slippage)
        self.quote_quality_variance = quote_quality_variance
        self._send_base_quantity = np.nan
        self._send_quote_quantity = np.nan

    @property
    def status(self):
        return {"active": self._activated,
                "quote_requested": self._quote_requested,
                "quote_received": self._quote_received,
                "quote_price": self._quote_price,
                "order_sent": self._sent,
                "order_cancel_at": self._cancel_at,
                "order_open": self._open,
                "open_order_matched": self._matched,
                "order_filled": self._filled
                }

    @property
    def has_order_info(self):
        return self._order_info != {}

    @property
    def is_strategy_ready(self):
        return not (self.is_active or self.is_filled)

    @property
    def is_sent_but_not_filled(self):
        return self._sent and not self._filled

    @property
    def strategy(self):
        return self._order_info.get("strategy", "")

    @property
    def display(self):
        keys = ["type", "side", "trade", "when", "price", "base_quantity", "quote_quantity",
                "profit_target", "price_movement", "strategy"]
        rounders = [["base_quantity", 6], ["price", 2], ["quote_quantity", 2],
                    ["profit_target", 4], ["price_movement", 2]]

        lite_order = {k: self._order_info.get(k, None) for k in keys}
        for target, decimals in rounders:
            if lite_order[target] is not None:
                lite_order[target] = round(lite_order[target], decimals)
        return str(lite_order)

    @property
    def open_order_id(self):
        return self._open_info if self._open_info is None else self._open_info[1]

    @property
    def open_order_info(self):
        return self._open_info

    def mark_active(self):
        self._activated = True

    def mark_filled(self):
        self._filled = True
        self._activated = False
        self._open = False
        self._open_info = None

    def as_fill(self, fee_amount, fee_token):
        fill_info = self.transaction.copy()
        fill_info["fee_amount"] = fee_amount
        fill_info["fee_token"] = fee_token
        return Fill(fill_info)

    def resubmit(self):
        self.reset()
        self.mark_active()

    def mark_send_quantities(self, base_quantity, quote_quantity):
        self._send_base_quantity = float(base_quantity)
        self._send_quote_quantity = float(quote_quantity)

    @property
    def send_base_quantity(self):
        return self._send_base_quantity

    def check_quote_quality(self, price_received, log, last_weighted_spot):
        order_info_when = self._order_info.get("when", "")
        order_info_price = self._order_info.get("price", np.nan)
        good_quote_received = False
        quote_price = np.nan
        log.info(f"Checking quote quality price received {price_received}, last_weighted_spot {last_weighted_spot}"
                 f" order {self.display}")
        if order_info_when == "above":
            good_quote_received = price_received > order_info_price * (1 - self.slippage)
            if good_quote_received:
                quote_price = min(price_received, last_weighted_spot) * (1 + self.order_fudge)
        elif order_info_when == "below":
            good_quote_received = price_received < order_info_price * (1 + self.slippage)
            if good_quote_received and price_received > last_weighted_spot * (1 - self.quote_quality_variance):
                quote_price = price_received * (1 - self.order_fudge)
            elif good_quote_received:
                quote_price = max(price_received, last_weighted_spot) * (1 - self.order_fudge)
        elif order_info_when == "any" and self.side == "b":
            good_quote_received = price_received < last_weighted_spot * (1 + self.quote_quality_variance)
            quote_price = price_received
        elif order_info_when == "any" and self.side == "s":
            good_quote_received = price_received > last_weighted_spot * (1 - self.quote_quality_variance)
            quote_price = price_received
        else:
            log.warning(f"Unrecognized order_info_when {order_info_when}")

        if good_quote_received:
            self._quote_received = True
            self._quote_price = quote_price
        log.info(f"Good Price = {good_quote_received}")
        return good_quote_received

    def reset(self):
        self._activated, self._quote_requested, self._quote_received, self._sent, self._open, \
            self._matched, self._filled = [False] * 7
        self._open_info = None
        self._quote_price = np.nan
        self._cancel_at = 0
        self._send_base_quantity = np.nan
        self._send_quote_quantity = np.nan


class PositionOrders(Orders):

    def __init__(self, log):
        super().__init__(log)
        self.position_open: PositionOrder = PositionOrder()
        self.position_close: PositionOrder = PositionOrder()
        self.stop_loss_order: PositionOrder = PositionOrder()

    @property
    def is_active(self):
        return self.active_order.is_active

    @property
    def is_ready_to_open(self):
        return self.position_open.is_strategy_ready and self.position_close.is_strategy_ready

    @property
    def open_order_ids(self):
        if self.position_open.is_sent_but_not_filled:
            return self.position_open.open_order_id
        elif self.position_close.is_sent_but_not_filled:
            return self.position_close.open_order_id
        else:
            return None

    @property
    def open_order_info(self):
        if self.position_open.is_sent_but_not_filled:
            return self.position_open.open_order_info
        elif self.position_close.is_sent_but_not_filled:
            return self.position_close.open_order_info
        else:
            return None

    @property
    def _order_types(self):
        return [self.position_open, self.position_close]

    @property
    def display(self):
        return [order.display for order in self._order_types]

    @property
    def active_position(self) -> PositionOrder:
        active_position = PositionOrder()
        for position in self._order_types:
            if position.is_active:
                active_position = position
        return active_position

    def set_by_orders_info(self, orders_info):
        if type(orders_info) != list:
            orders_info = [orders_info]
        for order_info in orders_info:
            if order_info.get("type", "") == "position open":
                self.position_open = PositionOrder(order_info)
            elif order_info.get("type", "") == "position close":
                self.position_close = PositionOrder(order_info)
            elif order_info.get("type", "") == "stop_loss":
                self.stop_loss_order = PositionOrder(order_info)
            else:
                raise "Cannot set invalid order info"

    def set_orders(self, orders):
        if type(orders) != list:
            orders = [orders]
        for order in orders:
            if order.type == "position open":
                self.position_open = order
            elif order.type == "position close":
                self.position_close = order
            elif order.type == "stop_loss":
                self.position_close = order
            else:
                raise Exception("Cannot set invalid order")

    def check_for_activation(self, last_weighted_spot_price):
        if last_weighted_spot_price != "" and self.active_order.is_activation_valid:
            for order in self._order_types:
                order.check_activation(last_weighted_spot_price)
                if order.is_active:
                    self._activate_order(order)
        return self.active_order.is_active

    @property
    def any_orders_blank(self):
        return pd.Series([order.type is None for order in self._order_types]).any()

    def match_open_order(self, open_info, log):
        if self.position_open.is_sent_but_not_filled:
            self.position_open.mark_open(open_info)
        elif self.position_close.is_sent_but_not_filled:
            self.position_close.mark_open(open_info)
        else:
            log.info("Open and Close position are both not sent or filled but received:\n"
                     f"{open_info}")

    def resubmit_active_position(self):
        self.active_position.resubmit()

    def reset_orders(self):
        for order in self._order_types:
            order.reset()
        self.active_order = PositionOrder()

    def clear_orders(self):
        self.stop_loss_order = PositionOrder()
        self.position_open = PositionOrder()
        self.position_close = PositionOrder()
        self.active_order = PositionOrder()


class WalletBalance:
    def __init__(self, market: Market = None, wallet_state=None):
        self._quote_bal = None
        self._base_bal = None
        self._market = market
        self.initial = None
        self._committed_balance = None
        if wallet_state is not None:
            self.update_wallet_state(wallet_state)

    @property
    def deltas(self):
        base_delta = self._base_bal - self.initial.get(self._market.base_symbol, 0) / \
            10 ** self._market.base_decimals
        quote_delta = self._quote_bal - self.initial.get(self._market.quote_symbol, 0) / \
            10 ** self._market.quote_decimals
        return base_delta, quote_delta

    @property
    def state_is_none(self):
        return self._committed_balance is None

    def base_quote_balance(self, last_weighted_spot):
        base_balance_quote = self._base_bal * last_weighted_spot
        return base_balance_quote

    def update_balances(self, base_bal=0, quote_bal=0):
        if base_bal == 0:
            self._base_bal = self._committed_balance.get(self._market.base_symbol, 0) / \
                             10 ** self._market.base_decimals
        else:
            self._base_bal = base_bal
        if quote_bal == 0:
            self._quote_bal = self._committed_balance.get(self._market.quote_symbol, 0) / \
                              10 ** self._market.quote_decimals
        else:
            self._quote_bal = quote_bal

    def update_market(self, market: Market):
        self._market = market

    def update_wallet_state(self, wallet_state):
        state = wallet_state["committed"]["balances"]
        if self.initial is None:
            self.initial = state
        self._committed_balance = state
        self.update_balances()

    @property
    def base(self):
        return self._base_bal

    @property
    def quote(self):
        return self._quote_bal


class Strategy:
    def __init__(self, raw_strategy):
        self._raw = raw_strategy
        self.name = raw_strategy["name"]
        self.description = raw_strategy["description"]
        self.base_loss_margin = raw_strategy["base_loss_margin"]
        self.expected_return = raw_strategy["expected_return"]
        self._directions = json.loads(raw_strategy["direction"])
        self._stop_win_threshold = raw_strategy["stop_win_threshold"]
        self._stop_win_loss_margin_pct = raw_strategy["stop_win_loss_margin_pct"]
        self._stop_win_curr_margin_pct = raw_strategy["stop_win_curr_margin_pct"]

    @staticmethod
    def decision(candle, direction, last_weighted_spot):
        target, op, cutoff = [direction[k] for k in ["target", "op", "cutoff"]]
        if cutoff == "last_weighted_spot":
            cutoff = last_weighted_spot
        engine = {"gt": candle[target] > cutoff,
                  "ge": candle[target] >= cutoff,
                  "lt": candle[target] < cutoff,
                  "le": candle[target] <= cutoff,
                  "le_lws": candle[target] <= last_weighted_spot,
                  "ge_lws": candle[target] >= last_weighted_spot}
        return engine.get(op, False)

    def is_active(self, last_weighted_spot, candle: pd.Series):
        decisions = pd.Series([self.decision(candle, direction, last_weighted_spot)
                               for direction in self._directions], dtype=np.float64)
        return decisions.all()

    @property
    def actions(self):
        return {"side": "b", "base_loss_margin": self.base_loss_margin, "base_profit_margin": self.expected_return,
                "stop_win_threshold": self._stop_win_threshold,
                "stop_win_loss_margin_pct": self._stop_win_loss_margin_pct,
                "stop_win_curr_margin_pct": self._stop_win_curr_margin_pct,
                "strategy": self.name}


class Strategies:
    def __init__(self, engine=None, candle_retention=10):
        self._strategies: list[Strategy] = []
        self._active = []
        self._candle_retention = candle_retention
        self._recent_candles = []
        if engine is not None:
            self.refresh_strategies(engine)

    def refresh_strategies(self, engine=None):
        raw_strategies = get_strategies(engine)
        for raw_strategy_id in raw_strategies.index:
            raw_strategy = raw_strategies.loc[raw_strategy_id]
            if raw_strategy["name"] not in self.strategy_names:
                self._strategies.append(Strategy(raw_strategy))

    @property
    def descriptions(self):
        return "\n".join([strat.description for strat in self._strategies])

    @property
    def strategy_names(self):
        return [strategy.name for strategy in self._strategies]

    @property
    def limit_strategy(self):
        return len(self._strategies) == 1

    def _process_candle(self, last_weighted_spot: float, candle: pd.Series = None):
        if candle is not None:
            self._recent_candles = self._recent_candles[-self._candle_retention:] + [candle]
        else:
            candle = self._recent_candles[-1]
        active_strategies = []
        for strategy in self._strategies:
            if strategy.is_active(last_weighted_spot, candle):
                active_strategies.append(strategy)
        self._active = active_strategies

    def get_actions(self, last_weighted_spot: float, candle=None):
        self._process_candle(last_weighted_spot, candle)
        best_strategy = None
        for strategy in self._active:
            if (best_strategy is None) or (strategy.expected_return > best_strategy.expected_return):
                best_strategy = strategy
        if best_strategy is not None:
            actions = best_strategy.actions
        else:
            actions = {"side": None, "base_loss_margin": None, "base_profit_margin": None, "stop_win_threshold": None,
                       "stop_win_loss_margin_pct": None, "stop_win_curr_margin_pct": None, "strategy": None}
        return actions

    def get_strategy_by_name(self, strategy_name) -> Strategy:
        for strategy in self._strategies:
            if strategy.name == strategy_name:
                return strategy


class BasePosition:
    """ A class that represents the position based trading strategy"""
    def __init__(self, logging: Logger, stop_win_thresh=0.0075, stop_win_curr_margin_pct=0.75,
                 stop_win_loss_margin_pct=0.5, simulation=True, test=False, **kwargs):
        """
        Constructs all the necessary attributes for the strategy object
        Parameters
        ----------
        :param logging: a Logger instance
        """
        self.trades = []
        self.current_profit_margin = 0
        self.current_margin_gain = 0
        self.log = logging
        self.liquidity: Liquidity = Liquidity(logger=self.log)
        self.current_quantile = "not set"
        self.max_buy_size = 5
        self.max_sell_size = 5
        self.simulation = simulation
        self.test = test
        self.market = Market()
        self.wallet_balance: WalletBalance = WalletBalance()
        self.fills: Fills = Fills()
        if simulation:
            self.get_first_trade(init_base_quantity=kwargs["init_base_quantity"], init_price=kwargs["init_price"],
                                 starting_side=kwargs["starting_side"])
        self.margin_strategy = None
        self.dev_engine = create_engine(os.environ.get("BOTTY_DEVELOPMENT_DB_URL"), echo=False)
        self.stage_engine = create_engine(os.environ.get("BOTTY_STAGING_DB_URL"), echo=False)
        engine = self.dev_engine if simulation else self.stage_engine
        self.strategies: Strategies = Strategies() if self.test else Strategies(engine)
        self.orders: PositionOrders = PositionOrders(logging)
        self.last_quote_balance = 0
        self.last_base_balance = 0
        self.base_profit_margin = 0.0015
        self.base_loss_margin = -0.0090
        self.profit_margin = np.nan
        self.loss_margin = -0.0090
        self.stop_win = np.nan
        self.stop_win_threshold = 0.0115
        self.stop_win_loss_margin_pct = 1
        self.stop_win_curr_margin_pct = 0.4
        self.recent_candles = DeepCandles()

    def get_first_trade(self, init_base_quantity, init_price, starting_side="b"):
        if starting_side == "s":
            last_fill = Fill({"market": "ETH-USDC",
                              "side": "s",
                              "trade": "ETH",
                              "price": init_price + 7.55,
                              "when": "below",
                              "base_quantity": init_base_quantity,
                              "type": "profit",
                              "fee_amount": self.market.base_fee,
                              "fee_token": "ETH"})
        else:
            last_fill = Fill({"market": "ETH-USDC",
                              "side": "b",
                              "trade": "USDC",
                              "price": init_price - 7.55,
                              "when": "above",
                              "base_quantity": init_base_quantity,
                              "type": "profit",
                              "fee_amount": self.market.quote_fee,
                              "fee_token": "USDC"})
        self.fills.add_fill(last_fill)

    @property
    def strategy_name(self):
        return f"Base Position"

    @property
    def strategy_description(self):
        return f"Strategy:\n\t" \
               f"Open a position when: \n" \
               f"{self.strategies.descriptions}"

    async def shut_it_down(self, message):
        if self.test:
            print(f"Bad Error: {message}")

    def transaction_writer(self, blob, blob_type):
        pass

    async def position_open_order(self, side, price, base_quantity=0, base_balance_=0, quote_balance_=0,
                                  order_fudge=0.000, slippage=0.000, opening_strategy="",
                                  quote_quality_variance=0.0020) -> PositionOrder:
        base, quote = self.market.base_quote
        self.wallet_balance.update_balances(base_bal=base_balance_, quote_bal=quote_balance_)

        base_fee = self.market.base_fee
        quote_fee = self.market.quote_fee
        price = float(price) if type(price) == str else price
        when = "any"
        if side == "b":
            trade = quote  # trading usdc (sell) for eth (buy)
            if base_quantity == 0:
                base_quantity = min(self.max_buy_size, (self.wallet_balance.quote - quote_fee) / price)
            else:
                base_quantity = float(base_quantity) if type(base_quantity) == str else base_quantity
                base_quantity = min(self.max_buy_size, (self.wallet_balance.quote - quote_fee) / price, base_quantity)
            if self.simulation:
                quote_quantity = base_quantity * price + quote_fee
                real_price = quote_quantity / base_quantity
            else:
                quote_quantity = (base_quantity - base_fee) * price
                real_price = quote_quantity / (base_quantity / price)

        elif side == "s":
            trade = base  # trading eth (sell) for usdc (buy)
            if base_quantity == 0:
                base_quantity = min(self.max_sell_size, self.wallet_balance.base - base_fee)
            else:
                base_quantity = min(self.max_sell_size, self.wallet_balance.base - base_fee, base_quantity)
            quote_quantity = (base_quantity - base_fee) * price
            real_price = quote_quantity / base_quantity
        else:
            await self.shut_it_down("side is neither buy nor sell")
            side, trade, price_to_profit, base_quantity, price_to_profit_no_fee, base_quantity_no_fee, \
                quote_quantity, real_price = 0, 0, 0, 0, 0, 0, 0, 0
        order_info = {"market": self.market.alias,
                      "side": side,
                      "trade": trade,
                      "when": when,
                      # for buys base_quantity is actually quote quantity on non-simulation
                      "base_quantity": base_quantity,
                      "price": price,
                      "base_quantity_with_fee": base_quantity,
                      "quote_quantity": quote_quantity,
                      "real_price": real_price,
                      "type": "position open",
                      "strategy": opening_strategy,
                      "profit_target": self.base_profit_margin,
                      "expected_profit_quote": quote_quantity * self.base_profit_margin,
                      "base_fee": base_fee}

        order = PositionOrder(order_info, order_fudge=order_fudge, slippage=slippage,
                              quote_quality_variance=quote_quality_variance)
        order.mark_active()
        self.orders.set_orders(order)
        self.transaction_writer(order.transaction, order.type + "_order")
        return order

    async def position_close_order(self, for_stop_loss=True, base_balance_=0, quote_balance_=0, order_fudge=0.0000,
                                   slippage=0.0000, quote_quality_variance=0.0020) \
            -> PositionOrder:
        base, quote = self.market.base_quote
        if self.simulation and base_balance_ == quote_balance_ == 0:
            pass
        elif self.test and base_balance_ == quote_balance_ == 0:
            pass
        else:
            self.wallet_balance.update_balances(base_bal=base_balance_, quote_bal=quote_balance_)

        if for_stop_loss:
            profit_margin = self.loss_margin
            trade_type = "position close"
        else:
            profit_margin = self.profit_margin
            trade_type = "profit"
        last = self.fills.last.fill
        last_price, last_side, last_base_quantity, last_quote_quantity =\
            [last[k] for k in ['real_price', 'side', 'base_quantity', 'quote_quantity']]
        base_fee = self.market.base_fee
        quote_fee = base_fee * last_price
        max_sell_base_quantity = min(self.max_sell_size, last_base_quantity, self.wallet_balance.base - base_fee)
        max_buy_quote_quantity = min(self.max_buy_size * last_price, last_quote_quantity,
                                     self.wallet_balance.quote - base_fee * last_price)
        if last_side == "s":
            side = "b"
            trade = quote  # trading usdc (sell) for eth (buy)
            when = "above" if for_stop_loss else "below"
            estimated_buy_price = self.liquidity.estimate_price_quote(max_buy_quote_quantity)
            price_denominator = (last_base_quantity + base_fee)
            price_calc = last_quote_quantity / price_denominator * (1 - profit_margin)
            price_to_profit = price_calc if for_stop_loss else min(price_calc, estimated_buy_price)
            price_calc_no_fee = last_price * (1 - profit_margin)
            price_to_profit_no_fee = price_calc_no_fee if for_stop_loss else \
                min(price_calc_no_fee, estimated_buy_price)
            if self.simulation:
                quote_quantity = min(last_quote_quantity, max_buy_quote_quantity)
                base_quantity = quote_quantity / price_to_profit - base_fee
                base_quantity_no_fee = quote_quantity / price_to_profit_no_fee
                real_price = quote_quantity / base_quantity
            else:
                base_quantity = min(last_quote_quantity, max_buy_quote_quantity)
                base_quantity_no_fee = min(last_quote_quantity, max_buy_quote_quantity)
                quote_quantity = (base_quantity / price_to_profit - base_fee) * price_to_profit
                real_price = quote_quantity / (base_quantity / price_to_profit)

        elif last_side == "b":
            side = "s"
            trade = base  # trading eth (sell) for usdc (buy)
            when = "below" if for_stop_loss else "above"
            estimated_sell_price = self.liquidity.estimate_price_base(max_sell_base_quantity, side='s')
            stop_loss_fudge = 0.01 / last_price if for_stop_loss else 0
            price_denominator = (last_base_quantity - self.market.base_fee - stop_loss_fudge)
            price_calc = (last_quote_quantity + quote_fee / (1 + profit_margin)) / \
                price_denominator * (1 + profit_margin)
            price_to_profit = price_calc if for_stop_loss else max(price_calc, estimated_sell_price)
            base_quantity = max_sell_base_quantity
            price_calc_no_fee = last_price * (1 + profit_margin)
            price_to_profit_no_fee = price_calc_no_fee if for_stop_loss else \
                max(price_calc_no_fee, estimated_sell_price)
            base_quantity_no_fee = max_sell_base_quantity
            quote_quantity = (base_quantity - base_fee) * price_to_profit
            real_price = quote_quantity / base_quantity
        else:
            await self.shut_it_down("side is neither buy nor sell")
            side, trade, price_to_profit, when, base_quantity, price_to_profit_no_fee, base_quantity_no_fee, \
                quote_quantity, real_price = \
                0, 0, 0, 0, 0, 0, 0, 0, 0
        if last_side == "s" and self.simulation is False:
            expected_profit_base = base_quantity_no_fee / price_to_profit * profit_margin
        else:
            expected_profit_base = base_quantity_no_fee * profit_margin

        order_info = {"market": self.market.alias,
                      "side": side,
                      "trade": trade,
                      "when": when,
                      # for buys base_quantity is actually quote quantity on non-simulation
                      "base_quantity": base_quantity_no_fee,
                      "base_quantity_no_fee": base_quantity_no_fee,
                      "price": price_to_profit_no_fee,
                      "price_no_fee": price_to_profit_no_fee,
                      "price_with_fee": price_to_profit,
                      "base_quantity_with_fee": base_quantity,
                      "quote_quantity": quote_quantity,
                      "real_price": real_price,
                      "type": trade_type,
                      "profit_target": profit_margin,
                      "expected_profit_base": expected_profit_base,
                      "expected_profit_quote": quote_quantity * profit_margin,
                      "base_fee": base_fee,
                      "price_movement": price_to_profit_no_fee - last_price,
                      "price_movement_no_fee": price_to_profit_no_fee - last_price,
                      "price_movement_with_fee": price_to_profit - last_price}

        order = PositionOrder(order_info, self.simulation, order_fudge=order_fudge, slippage=slippage,
                              quote_quality_variance=quote_quality_variance)
        self.orders.set_orders(order)
        self.transaction_writer(order.transaction, order.type + "_order")
        return order

    def convert_candle_to_fill(self, candle: pd.Series, last_fill):
        market = candle["market_id"]
        base, quote = self.market.base_quote
        if self.simulation:
            if last_fill["side"] == "b":
                price = candle["price_high"]
            else:
                price = candle["price_low"]
        else:
            price = candle["price_close"]
        new_fill = dict(candle.to_dict())
        new_fill["price"] = price
        new_fill["market"] = market
        new_fill["fill_status"] = "f"
        if last_fill["side"] == "b":
            new_fill["side"] = "s"
            fee = self.market.base_fee
            new_fill["fee_amount"] = fee
            new_fill["fee_token"] = base
            quote_quantity = (last_fill["base_quantity"]) * price
            base_quantity = quote_quantity / price
        else:
            new_fill["side"] = "b"
            fee = self.market.quote_fee
            new_fill["fee_amount"] = fee
            new_fill["fee_token"] = quote
            base_quantity = (last_fill["quote_quantity"]) / price
            quote_quantity = base_quantity * price
        new_fill["base_quantity"] = base_quantity
        new_fill["quote_quantity"] = quote_quantity
        return Fill(new_fill)

    async def calculate_profit(self, new_fill: Fill, old_fill: Fill, actual_trade=True):
        base, quote = self.market.base_quote
        old_fill_side, old_base_quantity, old_quote_quantity, old_price = \
            [old_fill.fill[k] for k in ('side', 'base_quantity', 'quote_quantity', 'real_price')]
        new_fill_side, new_base_quantity, new_quote_quantity, new_price = \
            [new_fill.fill[k] for k in ('side', 'base_quantity', 'quote_quantity', 'real_price')]
        base_fee = new_fill.get_fee(base, self.market.base_fee)
        quote_fee = new_fill.get_fee(quote, self.market.quote_fee)
        if new_fill_side == "s":
            base_profit = 0.0
            quote_revenue = (new_quote_quantity - min(old_quote_quantity,
                                                      self.max_buy_size * old_price))
            quote_cost = quote_fee
            quote_profit = quote_revenue - quote_cost
        elif new_fill_side == "b":
            base_revenue = new_base_quantity - min(old_base_quantity, self.max_sell_size)
            base_cost = base_fee
            base_profit = base_revenue - base_cost
            quote_profit = 0.0
        else:
            await self.shut_it_down("Impossible side for new fill")
            base_profit = 0
            quote_profit = 0
        base_return = base_profit / old_base_quantity
        quote_return = quote_profit / old_quote_quantity
        pct_return = 100 * (base_return if abs(base_profit) > 0 else quote_return)
        profit = {"pct_return": round(pct_return, 2),
                  base: base_profit,
                  quote: quote_profit,
                  "market": self.market.alias}
        if actual_trade and new_fill_side == "s":
            self.trades.append(profit)
        return profit

    async def update_current_profit(self, candle: pd.Series):
        last_fill = self.fills.last
        potential_fill = self.convert_candle_to_fill(candle, last_fill.fill)
        current_profit = await self.calculate_profit(potential_fill, last_fill, actual_trade=False)
        current_profit_margin = current_profit["pct_return"] / 100
        if current_profit_margin > self.current_margin_gain:
            self.current_margin_gain = current_profit_margin
            if current_profit_margin > self.stop_win_threshold:
                self.stop_win = max(current_profit_margin + self.base_loss_margin / self.stop_win_loss_margin_pct,
                                    current_profit_margin * self.stop_win_curr_margin_pct)
        if self.current_margin_gain < self.stop_win_threshold:
            self.stop_win = np.nan
        self.current_profit_margin = current_profit_margin

    async def process_volatility(self, candle=None):
        if self.simulation:
            loss = self.base_loss_margin
            profit = self.base_profit_margin
        else:
            profit, loss = self.base_profit_margin, self.base_loss_margin
        profit_different = self.profit_margin != profit + self.current_margin_gain
        loss_different = (self.loss_margin != loss + self.current_margin_gain and
                          self.loss_margin != self.stop_win)
        if profit is not None and not np.isnan(profit) and profit_different:
            self.profit_margin = profit + self.current_margin_gain
        if loss is not None and not np.isnan(loss) and loss_different:
            if np.isnan(self.stop_win):
                self.loss_margin = loss + self.current_margin_gain
            else:
                self.loss_margin = self.stop_win
        if (profit_different or loss_different) and self.orders.active_order.is_activation_valid and \
                self.orders.position_close.has_order_info:
            await self.position_close_order()

    async def is_trade_active(self, candle_: pd.Series, wallet) -> PositionOrder:
        candle = candle_
        await self.update_current_profit(candle)
        activated_order = PositionOrder()
        if self.orders.position_close.has_order_info:
            order = self.orders.position_close
            if order.check_activation(candle["price_low"]) or order.check_activation(candle["price_high"]):
                activated_order = order
        else:
            action = self.strategies.get_actions(candle["price_open"], candle)
            side, base_loss_margin, base_profit_margin, sw_thresh, sw_loss_margin_pct, sw_curr_margin_pct, strategy = [
                action[k] for k in ["side", "base_loss_margin", "base_profit_margin", "stop_win_threshold",
                                    "stop_win_loss_margin_pct", "stop_win_curr_margin_pct", "strategy"]
            ]
            if side in ["b", "s"]:
                self.base_loss_margin = base_loss_margin
                self.base_profit_margin = base_profit_margin
                self.stop_win_threshold = sw_thresh
                self.stop_win_loss_margin_pct = sw_loss_margin_pct
                self.stop_win_curr_margin_pct = sw_curr_margin_pct
                order = await self.position_open_order(side, candle["price_open"],
                                                       base_balance_=wallet.base_balance,
                                                       quote_balance_=wallet.quote_balance,
                                                       opening_strategy=strategy)
                self.orders.set_orders(order)
                activated_order = order
        price = candle["price_open"]
        self.liquidity.ingest_liquidity({"args": [1,
                                                  self.market.alias,
                                                  [["s", price, 1000],
                                                   ["b", price, 1000]]]})
        await self.process_volatility(candle=candle)
        return activated_order
