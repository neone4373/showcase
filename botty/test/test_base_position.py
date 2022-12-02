import asyncio
import json
import logging
import os
import unittest

import numpy as np
import pandas as pd
from pandas import Timestamp

from src.simulation_tools.positon_simulation import SimulationWallet
from src.zkbot.position_strategies.base_position import BasePosition, Strategy, WalletBalance, Market, Fill, Fills, \
    Strategies, PositionOrder, PositionOrders
from tests.fixtures.fixture_deep_ping_pong import position_open, position_close, position_open_buy, \
    position_open_sell
from tests.fixtures.new_last_fills import new_last_fill_2, new_last_fill_5, new_last_fill_6, new_last_fill_4, \
    new_last_fill_3
from tests.fixtures.sample_recent_candles import df_sub_select


class TestMarket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.market_info = {'alias': 'ETH-USDC',
                            'baseAsset': {'address': '0x0000000000000000000000000000000000000000',
                                          'decimals': 18,
                                          'enabledForFees': True,
                                          'id': 0,
                                          'name': 'Ethereum',
                                          'symbol': 'ETH'},
                            'baseAssetId': 0,
                            'baseAssetName': 'Ethereum',
                            'baseFee': 3.091e-05,
                            'displayName': 'ETH-USDC',
                            'id': 'kjkhjkhjjkhkjh',
                            'maxSize': 100,
                            'minSize': 0.0003,
                            'pricePrecisionDecimals': 2,
                            'quoteAsset': {'address': '0xeb8f08a975ab53e34d8a0330e0d34de942c95926',
                                           'decimals': 6,
                                           'enabledForFees': True,
                                           'id': 2,
                                           'name': 'USDC',
                                           'symbol': 'USDC'},
                            'quoteAssetId': 2,
                            'quoteAssetName': 'USD Coin',
                            'quoteFee': 0.006633,
                            'ChainId': 1000}
        self.mkt = Market()

    def test_update_market(self):
        assert self.mkt.alias is None
        assert np.isnan(self.mkt.base_fee)
        assert np.isnan(self.mkt.quote_fee)
        assert self.mkt.base_asset_id is None
        assert self.mkt.quote_asset_id is None
        assert self.mkt.base_decimals == 0
        assert self.mkt.base_symbol == ""
        assert self.mkt.quote_decimals == 0
        assert self.mkt.quote_symbol == ""
        self.mkt.update_market(self.market_info)

        assert self.mkt.alias == "ETH-USDC"
        assert self.mkt.base_fee == 3.091e-05
        assert self.mkt.quote_fee == 0.006633
        assert self.mkt.base_asset_id == 0
        assert self.mkt.quote_asset_id == 2
        assert self.mkt.base_decimals == 18
        assert self.mkt.base_symbol == "ETH"
        assert self.mkt.quote_decimals == 6
        assert self.mkt.quote_symbol == "USDC"

    def test_base_quote(self):
        self.mkt.update_market(self.market_info)
        base, quote = self.mkt.base_quote
        assert base == "ETH"
        assert quote == "USDC"

    def tearDown(self):
        print("tearDown")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")


class TestFill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.zk_id = "12345678"
        self.raw_fills = [
            [1000, 382, 'ETH-USDC', 's', 2923.07, 0.08003047, 'f',
             '', self.zk_id, '239i32904', 1 / 2923.07,
             "ETH"],
            [1000, 19, 'ETH-USDC', 'b', 2946.2, 0.02595, 'f',
             '', '1285557', self.zk_id, 1, "USDC"],
            [1000, 377, 'ETH-USDC', 's', 2937.33, 0.05096882, 'f',
             '', '1285557', '239i32904', 1 / 2937.33,
             "ETH"],
            [1000, 376, 'ETH-USDC', 'b', 2870.84, 0.7042, 'f',
             '', '230523', '239i32904', 1, "USDC"],
            [1000, 375, 'ETH-USDC', 's', 2864.06, 0.7059324499, 'f',
             '', '230523', '239i32904', None,
             ""],
            [1000, 382, 'ETH-USDC', 's', 2923.07, 0.08003047, 'm',
             '', self.zk_id, '239i32904', 1 / 2923.07,
             "ETH"]
        ]

    def test_append_fills(self):
        fill = Fill(self.raw_fills[0])
        assert fill.fill == {'base_fee': 0.00034210607340912124, 'base_quantity': 0.08003047, 'chain_id': 1000,
                             'fee_amount': 0.00034210607340912124, 'fee_token': 'ETH', 'fill_status': 'f',
                             'id': 382, 'maker_user_id': '239i32904', 'market': 'ETH-USDC', 'price': 2923.07,
                             'quote_quantity': 232.93466594290004, 'real_price': 2910.5747591248687, 'side': 's',
                             'taker_user_id': '12345678',
                             'tx_hash': ''}

        fill = Fill({'market': 'ETH-USDC', 'side': 's', 'trade': 'USDC', 'price': 1660,
                     'when': 'above', 'base_quantity': 0.25, 'type': 'profit', 'fee_amount': 0.00034210607340912124,
                     'fee_token': 'ETH', 'quote_quantity': 409.02675, "id": 1235,
                     'base_fee': 0.00017921201760314628, 'real_price': 1675})
        assert fill.fill == {'base_fee': 0.00034210607340912124,
                             'base_quantity': 0.25,
                             'fee_amount': 0.00034210607340912124,
                             'fee_token': 'ETH',
                             'id': 1235,
                             'market': 'ETH-USDC',
                             'price': 1660,
                             'quote_quantity': 414.43210391814085,
                             'real_price': 1657.7284156725634,
                             'side': 's',
                             'trade': 'USDC',
                             'type': 'profit',
                             'when': 'above'}

    def test_get_fee(self):
        fill = Fill(self.raw_fills[-2])
        assert fill.get_fee("ETH", 0.002) == 0.002

        fill = Fill(self.raw_fills[0])
        assert fill.get_fee("ETH", 0.002) == 0.00034210607340912124

    def test_id(self):
        fill = Fill(self.raw_fills[0])
        assert fill.id == 382

    def test_side(self):
        fill = Fill(self.raw_fills[0])
        assert fill.side == "s"

    def test_price(self):
        fill = Fill(self.raw_fills[0])
        assert fill.price == 2923.07

    def test_fee_amount(self):
        fill = Fill(self.raw_fills[0])
        assert fill.fee_amount == 0.00034210607340912124

    def test_fee_token(self):
        fill = Fill(self.raw_fills[0])
        assert fill.fee_token == "ETH"

    def test_status(self):
        fill = Fill(self.raw_fills[0])
        assert fill.status == "f"

    def test_price_base_quote_quantity(self):
        fill = Fill(self.raw_fills[0])
        assert fill.price_base_quote_quantity == [2923.07, 0.08003047, 232.93466594290004]

    def test_update_price(self):
        fill = Fill(self.raw_fills[0])
        old_fill = fill.fill.copy()
        fill.update_price(1700, "base_quantity", 0.25)
        assert fill.fill == {'base_fee': 0.00034210607340912124,
                             'base_quantity': 0.25,
                             'chain_id': 1000,
                             'fee_amount': 0.00034210607340912124,
                             'fee_token': 'ETH',
                             'fill_status': 'f',
                             'id': 382,
                             'maker_user_id': '239i32904',
                             'market': 'ETH-USDC',
                             'price': 1700,
                             'quote_quantity': 424.4184196752045,
                             'real_price': 1697.673678700818,
                             'side': 's',
                             'taker_user_id': '12345678',
                             'tx_hash': ''}

    def tearDown(self):
        print("tearDown")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")


class TestFills(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.fills = Fills()
        self.zk_id = "12345678"
        self.raw_fills = [
            [1000, 382, 'ETH-USDC', 's', 2923.07, 0.08003047, 'f',
             'aaasdasd239i32904', self.zk_id, '239i32904', 1 / 2923.07,
             "ETH"],
            [1000, 19, 'ETH-USDC', 'b', 2946.2, 0.02595, 'f',
             'aaasdasd239i32904', '1285557', self.zk_id, 1, "USDC"],
            [1000, 377, 'ETH-USDC', 's', 2937.33, 0.05096882, 'f',
             'aaasdasd239i32904', '1285557', '239i32904', 1 / 2937.33,
             "ETH"],
            [1000, 376, 'ETH-USDC', 'b', 2870.84, 0.7042, 'f',
             'aaasdasd239i32904', '230523', '239i32904', 1, "USDC"],
            [1000, 375, 'ETH-USDC', 's', 2864.06, 0.7059324499, 'm',
             'aaasdasd239i32904', '230523', '239i32904', None,
             ""]]

    def test_add_fills(self):
        self.fills.add_fill(Fill(self.raw_fills[0]))
        assert len(self.fills.fills) == 1

        self.fills.add_fill(Fill(self.raw_fills[0]))
        assert len(self.fills.fills) == 1

        self.fills.add_fill(Fill(self.raw_fills[1]))
        assert len(self.fills.fills) == 2

        self.fills.add_fill(Fill(self.raw_fills[4]))
        assert len(self.fills.fills) == 2
        assert self.fills.matched_fill.status == "m"

    def test_finalize_matched_fill(self):
        self.fills.add_fill(Fill(self.raw_fills[4]))
        self.fills.finalize_matched_fill([0, 1, 2, "0xHAT", 4, 0.1, "ETH"])
        assert self.fills.matched_fill.id == 0
        assert self.fills.last.id == 375

    def test_last(self):
        self.fills.add_fill(Fill(self.raw_fills[0]))
        assert self.fills.last.id == 382

    def tearDown(self):
        print("tearDown")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")


class TestPositionOrder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger('simple_example')
        logger.setLevel(logging.INFO)
        cls.logger = logger

    def setUp(self):
        self.po = PositionOrder(position_open)
        self.pc = PositionOrder(position_close)

    def test_status(self):
        assert self.po.status == {'active': False,
                                  'open_order_matched': False,
                                  'order_cancel_at': 0,
                                  'order_filled': False,
                                  'order_open': False,
                                  'order_sent': False,
                                  'quote_price': np.nan,
                                  'quote_received': False,
                                  'quote_requested': False}
        self.po.mark_active()
        self.po.mark_quote_requested()
        assert self.po.status == {'active': True,
                                  'open_order_matched': False,
                                  'order_cancel_at': 0,
                                  'order_filled': False,
                                  'order_open': False,
                                  'order_sent': False,
                                  'quote_price': np.nan,
                                  'quote_received': False,
                                  'quote_requested': True}

    def test_has_order_info(self):
        po = PositionOrder()
        assert not po.has_order_info

        po = PositionOrder(position_open)
        assert po.has_order_info

    def test_is_ready_to_activate(self):
        assert self.po.is_strategy_ready
        self.po.mark_active()
        assert not self.po.is_strategy_ready
        self.po.mark_filled()
        assert not self.po.is_strategy_ready
        self.po.reset()
        assert self.po.is_strategy_ready

    def test_is_sent_but_not_filled(self):
        assert not self.po.is_sent_but_not_filled
        self.po.mark_sent()
        assert self.po.is_sent_but_not_filled

    def test_strategy(self):
        po = PositionOrder(position_open)
        assert po.strategy == "fancy aroon"

    def test_display(self):
        po = PositionOrder(position_open)
        assert po.display == "{'type': 'position open', 'side': 's', 'trade': 'USDC', 'when': 'any', " \
                             "'price': 2462.0, 'base_quantity': 135.269805, 'quote_quantity': 134.28, " \
                             "'profit_target': -0.01, 'price_movement': 29.34, 'strategy': 'fancy aroon'}"

    def test_open_order_id(self):
        assert self.po.open_order_id is None
        self.po.mark_open([0, 345])
        assert self.po.open_order_id == 345

    def test_open_order_info(self):
        assert self.po.open_order_info is None
        self.po.mark_open([0, 345])
        assert self.po.open_order_info == [0, 345]

    def test_mark_active(self):
        assert not self.po.is_active
        self.po.mark_active()
        assert self.po.is_active

    def test_mark_filled(self):
        po = PositionOrder(position_open)
        assert not po.is_filled
        po._activated = True
        po._open = True
        po._open_info = {"things": " things!"}
        po.mark_filled()
        assert po.is_filled
        assert not po._activated
        assert not po._open
        assert not po._open_info

    def test_as_fill(self):
        po = PositionOrder(position_open)
        fill = po.as_fill(1, "USDC")
        assert fill.price_base_quote_quantity == [2462.0, 135.2698052, 330572.26040240005]

    def test_resubmit(self):
        self.po.mark_quote_requested()
        assert not self.po.is_send_quote_valid
        self.po.resubmit()
        assert self.po.is_send_quote_valid

    def test_mark_send_quantities(self):
        self.po.mark_send_quantities(12, 2200)
        assert self.po._send_base_quantity == 12
        assert self.po._send_quote_quantity == 2200

    def test_send_base_quantity(self):
        self.po.mark_send_quantities(12, 2200)
        assert self.po.send_base_quantity == 12

    def test_check_quote_quality__position_open(self):
        po_buy = PositionOrder(position_open_buy, quote_quality_variance=0.0020)
        # buy bad
        last_weighted_spot = 1222.57
        assert not po_buy.check_quote_quality(1307, self.logger, last_weighted_spot)
        assert not po_buy._quote_received
        assert np.isnan(po_buy._quote_price)
        # buy good
        last_weighted_spot = 1222.57
        assert po_buy.check_quote_quality(1224.76, self.logger, last_weighted_spot)
        assert po_buy._quote_received
        assert po_buy._quote_price == 1224.76
        po_sell = PositionOrder(position_open_sell, quote_quality_variance=0.0020)
        # sell bad
        last_weighted_spot = 1893.50
        assert not po_sell.check_quote_quality(1875, self.logger, last_weighted_spot)
        assert not po_sell._quote_received
        assert np.isnan(po_sell._quote_price)
        # sell good
        last_weighted_spot = 1893.50
        assert po_sell.check_quote_quality(1891.45, self.logger, last_weighted_spot)
        assert po_sell._quote_received
        assert po_sell._quote_price == 1891.45

    def test_check_quote_quality__position_close(self):
        close_position = PositionOrder()
        # 2567
        self.pc.order_fudge = 0.0000
        self.pc.quote_quality_variance = 0.0020
        last_weighted_spot = 2650
        assert not self.pc.check_quote_quality(2650, self.logger, last_weighted_spot)
        assert not self.pc._quote_received
        assert np.isnan(self.pc._quote_price)

        self.pc.reset()
        last_weighted_spot = 2560
        self.pc.order_fudge = 0.0020
        assert self.pc.check_quote_quality(2558, self.logger, last_weighted_spot)
        assert self.pc._quote_received
        assert self.pc._quote_price == 2552.884

        self.pc.reset()
        last_weighted_spot = 2560
        assert self.pc.check_quote_quality(2554, self.logger, last_weighted_spot)
        assert self.pc._quote_received
        assert self.pc._quote_price == 2554.88

        self.pc.reset()
        last_weighted_spot = 2680
        assert not self.pc.check_quote_quality(2650, self.logger, last_weighted_spot)
        assert not self.pc._quote_received
        assert np.isnan(self.pc._quote_price)

    def test_check_quote_quality_slippage(self):
        self.pc.slippage = 0.002
        last_weighted_spot = 2680
        assert not self.pc.check_quote_quality(2650, self.logger, last_weighted_spot)
        assert not self.pc._quote_received
        assert np.isnan(self.pc._quote_price)

        self.pc.reset()
        self.pc.quote_quality_variance = 0.0020
        last_weighted_spot = 2561
        self.pc.order_fudge = 0.0020
        assert self.pc.check_quote_quality(2558, self.logger, last_weighted_spot)
        assert self.pc._quote_received
        assert self.pc._quote_price == 2552.884

        self.pc.reset()
        last_weighted_spot = 2480
        assert self.pc.check_quote_quality(2563, self.logger, last_weighted_spot)
        assert self.pc._quote_received
        assert self.pc._quote_price == 2557.874

        self.pc.reset()
        last_weighted_spot = 2680
        assert not self.pc.check_quote_quality(2650, self.logger, last_weighted_spot)
        assert not self.pc._quote_received
        assert np.isnan(self.pc._quote_price)

    def test_reset(self):
        # should also reset send base and quote quantity
        self.po._activated, self.po._quote_requested, self.po._quote_received, self.po._sent, self.po._open, \
            self.po._matched, self.po._filled = [True] * 7
        self.po._open_info, self.po._fill_info = [{}] * 2
        self.po._send_base_quantity, self.po._send_quote_quantity = [23, 534]
        self.po._cancel_at = 333
        self.po.reset()
        assert self.po._activated is self.po._quote_requested is self.po._quote_received is\
               self.po._sent is self.po._open is self.po._matched is self.po._filled is False
        assert self.po._open_info is None
        assert self.po._cancel_at == 0
        assert self.po._order_info == position_open
        assert self.po.type == "position open"
        assert np.isnan(self.po._send_base_quantity)
        assert np.isnan(self.po._send_quote_quantity)

    def tearDown(self):
        print("tearDown")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")


class TestPositionOrders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger('simple_example')
        logger.setLevel(logging.INFO)

        cls.orders = PositionOrders(logger)

    def setUp(self):
        self.orders.clear_orders()
        self.blank = PositionOrder()
        self.po = PositionOrder(position_open)
        self.pc = PositionOrder(position_close)

    def test_is_active(self):
        assert not self.orders.is_active

        self.pc._activated = True
        self.orders._activate_order(self.pc)
        assert self.orders.is_active

    def test_is_ready_to_open(self):
        assert self.orders.is_ready_to_open
        self.orders.position_open.mark_active()
        assert not self.orders.is_ready_to_open
        self.orders.reset_orders()
        assert self.orders.is_ready_to_open
        self.orders.position_close.mark_active()
        assert not self.orders.is_ready_to_open
        self.orders.clear_orders()
        assert self.orders.is_ready_to_open

    def test_open_order_ids(self):
        assert self.orders.open_order_ids is None
        self.po._sent = True
        self.po.mark_open([1, 234])
        self.orders.set_orders(self.po)
        assert self.orders.open_order_ids == 234

    def test_open_order_info(self):
        assert self.orders.open_order_info is None
        self.po._sent = True
        self.po.mark_open([1, 234])
        self.orders.set_orders(self.po)
        assert self.orders.open_order_info == [1, 234]
        self.orders.position_open.mark_filled()
        assert self.orders.open_order_info is None
        self.pc._sent = True
        self.pc.mark_open([1, 235])
        self.orders.set_orders(self.pc)
        assert self.orders.open_order_info == [1, 235]

    def test_display(self):
        assert self.orders.display == ["{'type': None, 'side': None, 'trade': None, 'when': None, 'price': None, "
                                       "'base_quantity': None, 'quote_quantity': None, 'profit_target': None, "
                                       "'price_movement': None, 'strategy': None}",
                                       "{'type': None, 'side': None, 'trade': None, 'when': None, 'price': None, "
                                       "'base_quantity': None, 'quote_quantity': None, 'profit_target': None, "
                                       "'price_movement': None, 'strategy': None}"]

        self.orders.set_orders([self.po])
        assert self.orders.display == ["{'type': 'position open', 'side': 's', 'trade': 'USDC', 'when': 'any', "
                                       "'price': 2462.0, 'base_quantity': 135.269805, 'quote_quantity': 134.28, "
                                       "'profit_target': -0.01, 'price_movement': 29.34, 'strategy': 'fancy aroon'}",
                                       "{'type': None, 'side': None, 'trade': None, 'when': None, 'price': None, "
                                       "'base_quantity': None, 'quote_quantity': None, 'profit_target': None, "
                                       "'price_movement': None, 'strategy': None}"]

    def test_active_position(self):
        assert not self.orders.active_position.is_send_quote_valid
        assert not self.orders.active_position.type
        self.po._activated = True
        self.orders.set_orders(self.po)
        assert self.orders.active_position.type == "position open"

    def test_set_by_orders_info(self):
        self.orders.set_by_orders_info(position_open)
        assert self.orders.position_close.type is None
        assert self.orders.position_open.type == "position open"

    def test_set_orders(self):
        self.orders.set_orders(self.pc)
        assert self.orders.position_close.type == "position close"
        assert self.orders.position_open.type is None

    def test_check_for_activation(self):
        assert not self.orders.check_for_activation("")
        assert not self.orders.check_for_activation(2400)

        self.orders.set_orders([self.pc])
        # price 2462, any, sell
        self.orders.reset_orders()
        assert self.orders.check_for_activation(2564)
        assert self.orders.active_order.type == "position close"
        self.orders.reset_orders()
        assert not self.orders.check_for_activation(2600)
        assert self.orders.active_order.type is None

    def test_any_orders_blank(self):
        self.orders.set_orders(self.po)
        assert self.orders.any_orders_blank
        self.orders.set_orders(self.pc)
        assert not self.orders.any_orders_blank

    def test_match_open_order(self):
        logger = logging.getLogger('simple_example')
        logger.setLevel(logging.INFO)
        new_open_order = [1000, 644, 'ETH-USDC', 's', 3396.76583855155, 0.0485319,
                          164.8515, 1645541019, '1285383', 'o', 0.0485319]
        self.orders.position_close.mark_sent()
        assert not self.orders.position_open.is_sent_but_not_filled
        assert self.orders.position_close.is_sent_but_not_filled
        self.orders.match_open_order(new_open_order, logger)
        assert self.orders.position_close._open_info == new_open_order

    def test_resubmit_active_position(self):
        self.po.mark_active()
        self.po.mark_quote_requested()
        self.orders.set_orders(self.po)
        assert not self.orders.position_open.is_send_quote_valid
        self.orders.resubmit_active_position()
        assert self.orders.position_open.is_send_quote_valid

    def test_reset_orders(self):
        self.po._activated, self.po._quote_requested, self.po._quote_received, self.po._sent, self.po._open, \
            self.po._matched, self.po._filled = [True] * 7
        self.po._open_info, self.po._fill_info = [{}] * 2
        self.po._cancel_at = 333
        self.orders.set_orders([self.po])
        assert self.orders.position_open.display == \
               "{'type': 'position open', 'side': 's', 'trade': 'USDC', 'when': 'any', " \
               "'price': 2462.0, 'base_quantity': 135.269805, 'quote_quantity': 134.28, " \
               "'profit_target': -0.01, 'price_movement': 29.34, 'strategy': 'fancy aroon'}"
        self.orders.reset_orders()
        assert not self.orders.position_open.is_filled

    def test_clear_orders(self):
        self.orders.set_orders([self.po, self.pc])
        assert self.orders.position_open.type == "position open"
        assert self.orders.position_close.type == "position close"
        self.orders.clear_orders()
        assert self.orders.position_open.type is self.orders.position_close.type is None

    def tearDown(self):
        print("tearDown")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")


class TestWalletBalance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mkt = Market({'alias': 'ETH-USDC',
                          'baseAsset': {'address': '0x0000000000000000000000000000000000',
                                        'decimals': 18,
                                        'enabledForFees': True,
                                        'id': 0,
                                        'name': 'Ethereum',
                                        'symbol': 'ETH'},
                          'baseAssetId': 0,
                          'baseAssetName': 'Ethereum',
                          'baseFee': 3.091e-05,
                          'displayName': 'ETH-USDC',
                          'id': '',
                          'maxSize': 100,
                          'minSize': 0.0003,
                          'pricePrecisionDecimals': 2,
                          'quoteAsset': {'address': '0xeb8f08a975ab53e0330e0d34de942c95926',
                                         'decimals': 6,
                                         'enabledForFees': True,
                                         'id': 2,
                                         'name': 'USDC',
                                         'symbol': 'USDC'},
                          'quoteAssetId': 2,
                          'quoteAssetName': 'USD Coin',
                          'quoteFee': 0.006633,
                          'ChainId': 1000})

    def setUp(self):
        self.wal = WalletBalance(self.mkt,
                                 {'address': "0x", 'id': 1, 'account_type': 1, 'depositing': 6,
                                  'committed': {'balances': {'ETH': 190705010000000000,
                                                             'USDC': 308062780}},
                                  'verified': {'balances': {'ETH': 190705010000000000,
                                                            'USDC': 308062780}}})

    def test_state_is_none(self):
        wal = WalletBalance()
        assert wal.state_is_none
        assert not self.wal.state_is_none

    def test_update_wallet_state(self):
        wal = WalletBalance()
        assert wal.initial is None
        assert wal.state_is_none
        wal.update_market(self.mkt)
        wal.update_wallet_state({'address': "0x", 'id': 1, 'account_type': 1, 'depositing': 6,
                                 'committed': {'balances': {'ETH': 190705010000000000,
                                                            'USDC': 308062780}},
                                 'verified': {'balances': {'ETH': 190705010000000000,
                                                           'USDC': 308062780}}})
        assert wal.initial == {'ETH': 190705010000000000, 'USDC': 308062780}
        assert wal._base_bal == 0.19070501
        assert wal._quote_bal == 308.06278

    def test_deltas(self):
        base_delta, quote_delta = self.wal.deltas
        assert base_delta == 0
        assert quote_delta == 0

        self.wal.update_wallet_state({'address': "0x", 'id': 1, 'account_type': 1, 'depositing': 6,
                                      'committed': {'balances': {'ETH': 2190705010000000000,
                                                                 'USDC': 258062780}},
                                      'verified': {'balances': {'ETH': 190705010000000000,
                                                                'USDC': 308062780}}})
        base_delta, quote_delta = self.wal.deltas
        assert base_delta == 1.9999999999999998
        assert quote_delta == -50

    def test_quote_balance(self):
        base_balance_quote = self.wal.base_quote_balance(1500)
        assert base_balance_quote == 286.057515

    def tearDown(self):
        print("tearDown")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")


class TestStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = Strategy(pd.Series(
            {"name": "basic ppo",
             "description": "nah",
             "direction": json.dumps([
                 {"target": "momentum_ppo", "op": "gt", "cutoff": 1},
                 {"target": "momentum_ppo", "op": "gt", "cutoff": 5}]),
             "base_loss_margin": -0.0040,
             "expected_return": 0.0020,
             "stop_win_threshold": 0.01,
             "stop_win_loss_margin_pct": 0.75,
             "stop_win_curr_margin_pct": 1.00}
        ))

    def setUp(self):
        self.candle_1 = pd.Series({"momentum_ppo": 0})
        self.candle_2 = pd.Series({"momentum_ppo": 3})
        self.candle_3 = pd.Series({"momentum_ppo": 10})

    def test_decision(self):
        assert not self.st.decision(self.candle_1, self.st._directions[0], 1234)
        assert self.st.decision(self.candle_2, self.st._directions[0], 1234)
        st = Strategy(pd.Series(
            {"name": "basic ppo",
             "description": "nah",
             "direction": json.dumps([
                 {"target": "broke_yesterdays_high", "op": "lt", "cutoff": 1},
                 {"target": "broke_last_hours_high", "op": "gt", "cutoff": 0},
                 {"target": "price_close", "op": "le_lws", "cutoff": "last_weighted_spot"}
             ]),
             "base_loss_margin": -0.0040,
             "expected_return": 0.0020,
             "stop_win_threshold": 0.01,
             "stop_win_loss_margin_pct": 0.75,
             "stop_win_curr_margin_pct": 1.00}
        ))
        candle_1 = pd.Series({"broke_yesterdays_high": False,
                              "broke_last_hours_high": True,
                              "price_close": 1238})
        assert st.decision(candle_1, st._directions[0], 1234)
        assert st.decision(candle_1, st._directions[1], 1234)
        assert not st.decision(candle_1, st._directions[2], 1234)
        assert st.decision(candle_1, st._directions[2], 1241)

    def test_is_active(self):
        assert not self.st.is_active(1234, self.candle_1)
        assert not self.st.is_active(1234, self.candle_2)
        assert self.st.is_active(1234, self.candle_3)

    def test_actions(self):
        assert self.st.actions == {'base_loss_margin': -0.004, 'side': 'b',
                                   "base_profit_margin": 0.0020,
                                   'stop_win_curr_margin_pct': 1.0,
                                   'stop_win_loss_margin_pct': 0.75,
                                   'stop_win_threshold': 0.01,
                                   "strategy": "basic ppo"}

    def tearDown(self):
        print("tearDown")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")


class TestStrategies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.strategies = Strategies()
        self.candle_1 = pd.Series({"momentum_ppo": 0})
        self.candle_2 = pd.Series({"momentum_ppo": 3})
        self.candle_3 = pd.Series({"momentum_ppo": 10})

    def test_refresh_strategies(self):
        self.strategies.refresh_strategies()
        assert self.strategies._strategies[0].name == "basic ppo"

    def test_descriptions(self):
        self.strategies.refresh_strategies()
        desc = self.strategies.descriptions
        assert desc == "nah"

    def test_strategy_names(self):
        self.strategies.refresh_strategies()
        assert self.strategies.strategy_names == ["basic ppo"]

    def test__process_candle(self):
        self.strategies.refresh_strategies()
        self.strategies._process_candle(1234, self.candle_1)
        assert self.strategies._recent_candles == [self.candle_1]
        assert self.strategies._active == []

        self.strategies._process_candle(1234, self.candle_3)
        assert self.strategies._recent_candles == [self.candle_1, self.candle_3]
        assert self.strategies._active[0].name == 'basic ppo'

    def test_get_actions(self):
        self.strategies.refresh_strategies()
        actions = self.strategies.get_actions(1234, self.candle_1)
        assert actions == {'base_loss_margin': None,
                           'base_profit_margin': None,
                           'side': None,
                           'stop_win_curr_margin_pct': None,
                           'stop_win_loss_margin_pct': None,
                           'stop_win_threshold': None,
                           'strategy': None}

        self.strategies.refresh_strategies()
        actions = self.strategies.get_actions(1234, self.candle_3)
        assert actions == {'base_loss_margin': -0.004, 'base_profit_margin': 0.001,
                           'side': 'b',
                           'stop_win_curr_margin_pct': 1.0,
                           'stop_win_loss_margin_pct': 0.75,
                           'stop_win_threshold': 0.01, 'strategy': 'basic ppo'}

        fancy_ppo = Strategy(pd.Series(
            {"name": "fancy ppo",
             "description": "nah",
             "direction": json.dumps([
                 {"target": "momentum_ppo", "op": "gt", "cutoff": 5},
                 {"target": "momentum_ppo", "op": "gt", "cutoff": 5}]),
             "base_loss_margin": -0.0140,
             "expected_return": 0.0120,
             'stop_win_curr_margin_pct': 1.0,
             'stop_win_loss_margin_pct': 0.75,
             'stop_win_threshold': 0.01}
        ))
        self.strategies._strategies.append(fancy_ppo)
        actions = self.strategies.get_actions(1234, self.candle_3)
        assert actions == {'base_loss_margin': -0.014, 'base_profit_margin': 0.012,
                           'side': 'b',
                           'stop_win_curr_margin_pct': 1.0,
                           'stop_win_loss_margin_pct': 0.75,
                           'stop_win_threshold': 0.01, "strategy": "fancy ppo"}

    def test_get_strategy_by_name(self):
        self.strategies.refresh_strategies()
        strategy = self.strategies.get_strategy_by_name("basic ppo")
        assert strategy.name == "basic ppo"
        strategy = self.strategies.get_strategy_by_name("fancy ppo")
        assert not strategy

    def tearDown(self):
        print("tearDown")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")


class TestBasePosition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger = logging.getLogger('simple_example')
        logger.setLevel(logging.INFO)
        cls.bp = BasePosition(logger, simulation=False, test=True)

    def setUp(self):
        self.bp.test = True
        self.bp.simulation = False
        self.market = "ETH-USDC"
        self.bp.max_buy_size = 5
        self.bp.max_sell_size = 5
        self.bp.market = Market({'alias': 'ETH-USDC',
                                 'baseAsset': {'address': '0x0000000000000000000000000000000000000000',
                                               'decimals': 18,
                                               'enabledForFees': True,
                                               'id': 0,
                                               'name': 'Ethereum',
                                               'symbol': 'ETH'},
                                 'baseAssetId': 0,
                                 'baseAssetName': 'Ethereum',
                                 'baseFee': 3.091e-05,
                                 'displayName': 'ETH-USDC',
                                 'id': '',
                                 'maxSize': 100,
                                 'minSize': 0.0003,
                                 'pricePrecisionDecimals': 2,
                                 'quoteAsset': {'address': '0xeb8f08a975ab53e34d8a0330e0d34de942c95926',
                                                'decimals': 6,
                                                'enabledForFees': True,
                                                'id': 2,
                                                'name': 'USDC',
                                                'symbol': 'USDC'},
                                 'quoteAssetId': 2,
                                 'quoteAssetName': 'USD Coin',
                                 'quoteFee': 0.006633,
                                 'ChainId': 1000})
        self.bp.wallet_balance = WalletBalance(self.bp.market,
                                               {'address': "0x", 'id': 1, 'account_type': 1, 'depositing': 6,
                                                'committed': {'balances': {'ETH': 190705010000000000,
                                                                           'USDC': 308062780}},
                                                'verified': {'balances': {'ETH': 190705010000000000,
                                                                          'USDC': 308062780}}})

        self.zk_id = ""
        self.bp.profit_margin = 0.01
        self.bp.base_profit_margin = 0.01
        self.bp.loss_margin = -0.01
        self.bp.base_loss_margin = -0.01
        self.bp.current_margin_gain = 0
        self.bp.wallet_state = {}
        self.bp.last_base_balance = 5
        self.bp.last_quote_balance = 15000
        self.bp.target_markets = [self.market]
        self.bp.zk_id = self.zk_id
        self.bp.fills = Fills()
        self.bp.fills.add_fill(Fill({'market': 'ETH-USDC', 'side': 'b', 'trade': 'USDC', 'price': 1634.94,
                                     'when': 'above', 'base_quantity': 0.25, 'type': 'profit', "id": 1232,
                                     'fee_amount': 0.293, 'fee_token': 'USDC', 'quote_quantity': 409.02675,
                                     'base_fee': 0.00017921201760314628, 'real_price': 1636.107}))
        self.candle_1 = {'market_id': 'ETH-USDC',
                         'traded_at': Timestamp('2021-02-03 18:35:00'),
                         'price_low': 1640.95,
                         'price_high': 1644.02,
                         'price_open': 1641.14,
                         'price_close': 1643.68,
                         'volume': 7.9705567,
                         'traded_on': '21-02-03',
                         'pm': 0.005,
                         'sl': -0.005}
        self.bp.stop_win = np.nan

    def test_strategy_name(self):
        assert self.bp.strategy_name == 'Base Position'

    def test_strategy_description(self):
        assert self.bp.strategy_description[:126] == \
               'Strategy:\n' \
               '\tOpen a position when: \n'

    def test_position_open_order(self):
        self.bp.simulation = True
        order = asyncio.run(self.bp.position_open_order("b", 2400, base_balance_=0.5, quote_balance_=1400))
        assert order.display == "{'type': 'position open', 'side': 'b', 'trade': 'USDC', 'when': 'any', " \
                                "'price': 2400, 'base_quantity': 0.583331, 'quote_quantity': 1400.0, " \
                                "'profit_target': 0.01, 'price_movement': None, 'strategy': ''}"

        order = asyncio.run(self.bp.position_open_order("s", 2400, base_balance_=0.2, quote_balance_=1400))
        assert order.display == "{'type': 'position open', 'side': 's', 'trade': 'ETH', 'when': 'any', " \
                                "'price': 2400, 'base_quantity': 0.199969, 'quote_quantity': 479.85, " \
                                "'profit_target': 0.01, 'price_movement': None, 'strategy': ''}"

        order = asyncio.run(self.bp.position_open_order("b", 2400, base_balance_=0.5, quote_balance_=400))
        assert order.display == "{'type': 'position open', 'side': 'b', 'trade': 'USDC', 'when': 'any', " \
                                "'price': 2400, 'base_quantity': 0.166664, 'quote_quantity': 400.0, " \
                                "'profit_target': 0.01, 'price_movement': None, 'strategy': ''}"
        assert order.is_active

        self.bp.simulation = False
        order = asyncio.run(self.bp.position_open_order("b", 2400, base_balance_=0.5, quote_balance_=400))
        assert order.display == "{'type': 'position open', 'side': 'b', 'trade': 'USDC', 'when': 'any', " \
                                "'price': 2400, 'base_quantity': 0.166664, 'quote_quantity': 399.92, " \
                                "'profit_target': 0.01, 'price_movement': None, 'strategy': ''}"
        assert order.is_active

    def test_position_close_order_sim(self):
        # profit, buy
        self.bp.last_price = {"ETH-USDC": 2955.32}
        self.bp.last_weighted_spot = {"ETH-USDC": 2955.32}
        self.bp.market.base_fee = 1 / self.bp.last_weighted_spot["ETH-USDC"]
        self.bp.market.quote_fee = 1
        self.bp.fills.add_fill(Fill({'chain_id': 1000, 'id': 98243978423, 'market': 'ETH-USDC', 'side': 's',
                                     'price': 2955.32, 'base_quantity': 0.04611, 'fill_status': 'f',
                                     'tx_hash': 'cookies', 'taker_user_id': '123', "maker_user_id": self.zk_id,
                                     'fee_amount': 0.0003383728327220064, 'fee_token': 'USDC',
                                     'quote_quantity': 135.2698052, 'base_fee': 0.0003383728327220064,
                                     'real_price': 2933.6327304272395}))

        self.bp.simulation = True
        self.bp.wallet_balance.update_wallet_state({'committed': {'balances': {'ETH': 1e18, 'USDC': 2500e6}}})
        self.bp.profit_margin = 0.01
        self.bp.base_profit_margin = 0.01


        # stop loss, buy
        self.bp.loss_margin = -0.01
        self.bp.base_loss_margin = -0.01
        last = self.bp.fills.last.fill.copy()
        new_quote = asyncio.run(self.bp.position_close_order(for_stop_loss=True,
                                                             base_balance_=1, quote_balance_=2500))
        assert new_quote.transaction["side"] == "b"
        assert new_quote.transaction["trade"] == "USDC"
        assert round(new_quote.transaction["price"], 2) == 2941.38
        assert round(new_quote.transaction["base_quantity"], 6) == 0.04565
        assert self.bp.orders.position_close.transaction["when"] == "above"
        profit = asyncio.run(self.bp.calculate_profit(new_quote.as_fill(self.bp.market.quote_fee, "USDC"),
                                                      self.bp.fills.last))

        # stop loss, sell
        self.bp.last_weighted_spot = {"ETH-USDC": 2982.93}
        self.bp.market.base_fee = 1 / self.bp.last_weighted_spot["ETH-USDC"]
        self.bp.market.quote_fee = 1
        self.bp.fills.add_fill(Fill(new_last_fill_2))
        last = self.bp.fills.last.fill.copy()
        new_quote = asyncio.run(self.bp.position_close_order(for_stop_loss=True,
                                                             base_balance_=1, quote_balance_=2500))
        assert new_quote.transaction["side"] == "s"
        assert new_quote.transaction["trade"] == "ETH"
        assert round(new_quote.transaction["price"], 2) == 2997.51
        assert round(new_quote.transaction["base_quantity"], 6) == 0.04538
        assert self.bp.orders.position_close.transaction["when"] == "below"
        profit = asyncio.run(self.bp.calculate_profit(new_quote.as_fill(self.bp.market.base_fee, "ETH"),
                                                      self.bp.fills.last))
        assert (profit["ETH"] + profit["USDC"] / new_quote.transaction["price"]) / last["base_quantity"] >= \
               self.bp.loss_margin

    def test_position_close_order(self):
        # profit, buy
        self.bp.last_price = {"ETH-USDC": 2955.32}
        self.bp.last_weighted_spot = {"ETH-USDC": 2955.32}
        self.bp.market.base_fee = 1 / self.bp.last_weighted_spot["ETH-USDC"]
        self.bp.market.quote_fee = 1
        self.bp.fills.add_fill(Fill({'chain_id': 1000, 'id': 121319, 'market': 'ETH-USDC', 'side': 's',
                                     'price': 2955.32, 'base_quantity': 0.04611, 'fill_status': 'f',
                                     'tx_hash': 'cookies', 'taker_user_id': '123', "maker_user_id": self.zk_id,
                                     'fee_amount': 0.0003383728327220064, 'fee_token': 'USDC',
                                     'quote_quantity': 135.2698052, 'base_fee': 0.0003383728327220064,
                                     'real_price': 2933.6327304272395}))

        self.bp.wallet_state = {'committed': {'balances': {'ETH': 1e18, 'USDC': 2500e6}}}
        self.bp.profit_margin = 0.01
        last = self.bp.fills.last.fill.copy()

        # stop loss, buy
        self.bp.loss_margin = -0.01
        last = self.bp.fills.last.fill.copy()
        new_quote = asyncio.run(self.bp.position_close_order(for_stop_loss=True,
                                                             base_balance_=1, quote_balance_=2500))
        assert new_quote.transaction["side"] == "b"
        assert new_quote.transaction["trade"] == "USDC"
        assert round(new_quote.transaction["price"], 2) == 2962.97
        assert round(new_quote.transaction["base_quantity"], 6) == 135.269805
        assert self.bp.orders.position_close.transaction["when"] == "above"
        assert new_quote.transaction["base_quantity"] == (last["base_quantity"] -
                                                          self.bp.market.base_fee) * \
               last["price"]
        new_quote.transaction["base_quantity"] = new_quote.transaction["base_quantity"] / new_quote.transaction["price"]
        assert round((last["real_price"] - new_quote.transaction["price"]) / last["real_price"],
                     6) == self.bp.loss_margin

        # stop loss, sell
        self.bp.last_weighted_spot = {"ETH-USDC": 2982.93}
        self.bp.market.base_fee = 1 / self.bp.last_weighted_spot["ETH-USDC"]
        self.bp.market.quote_fee = 1
        self.bp.fills.add_fill(Fill(new_last_fill_2))
        last = self.bp.fills.last.fill.copy()
        new_quote = asyncio.run(self.bp.position_close_order(for_stop_loss=True,
                                                             base_balance_=1, quote_balance_=2500))
        assert new_quote.transaction["side"] == "s"
        assert new_quote.transaction["trade"] == "ETH"
        assert round(new_quote.transaction["price"], 2) == 2953.11
        assert round(new_quote.transaction["base_quantity"], 6) == 0.04538
        assert self.bp.orders.position_close.transaction["when"] == "below"
        profit = asyncio.run(self.bp.calculate_profit(new_quote.as_fill(self.bp.market.base_fee, "ETH"),
                                                      self.bp.fills.last))

    def test_next_trade_capped_sim(self):
        self.bp.market.base_fee = 1 / 2955.32
        self.bp.market.quote_fee = 1
        self.bp.last_price = {"ETH-USDC": 2955.32}
        self.bp.last_weighted_spot = {"ETH-USDC": 2955.32}
        self.bp.fills.add_fill(Fill(new_last_fill_5))

        self.bp.simulation = True
        self.bp.max_sell_size = 0.301
        self.bp.max_buy_size = 0.30
        self.bp.loss_margin = -0.01
        self.bp.profit_margin = 0.01
        self.bp.last_weighted_spot = {"ETH-USDC": 2982.93}
        self.bp.wallet_balance.update_balances({'committed': {'balances': {'ETH': 1e18, 'USDC': 2500e6}}})

        # stop loss, sell, sell capped
        self.bp.fills.add_fill(Fill(new_last_fill_6))
        self.bp.last_weighted_spot = {"ETH-USDC": 2601.70466}
        self.bp.market.base_fee = 1 / self.bp.last_weighted_spot["ETH-USDC"]
        self.bp.market.quote_fee = 1
        # self.bp.loss_margin = -0.0002
        new_quote = asyncio.run(self.bp.position_close_order(for_stop_loss=True,
                                                             base_balance_=1, quote_balance_=2500))
        assert new_quote.transaction["side"] == "s"
        assert new_quote.transaction["trade"] == "ETH"
        assert round(new_quote.transaction["price"], 2) == 2580.88
        assert round(new_quote.transaction["base_quantity"], 6) == 0.301

        # stop loss, sell, sell capped, small trigger
        self.bp.fills.add_fill(Fill(new_last_fill_6))
        self.bp.last_weighted_spot = {"ETH-USDC": 2601.70466}
        self.bp.market.base_fee = 1 / self.bp.last_weighted_spot["ETH-USDC"]
        self.bp.market.quote_fee = 1
        self.bp.loss_margin = -0.0002
        new_quote = asyncio.run(self.bp.position_close_order(for_stop_loss=True,
                                                             base_balance_=1, quote_balance_=2500))
        assert new_quote.transaction["side"] == "s"
        assert new_quote.transaction["trade"] == "ETH"
        assert round(new_quote.transaction["price"], 2) == 2606.4
        assert round(new_quote.transaction["base_quantity"], 6) == 0.301

    def test_next_trade_price_capped_sim(self):
        self.bp.last_price = {"ETH-USDC": 2955.32}
        self.bp.last_weighted_spot = {"ETH-USDC": 2955.32}
        self.bp.market.base_fee = 1 / self.bp.last_weighted_spot["ETH-USDC"]
        self.bp.market.quote_fee = 1
        self.bp.fills.add_fill(Fill({'chain_id': 1000, 'id': 19, 'market': 'ETH-USDC', 'side': 's',
                                     'price': 2955.32, 'base_quantity': 0.04611, 'fill_status': 'f',
                                     'tx_hash': 'cookies', 'taker_user_id': '123', "maker_user_id": self.zk_id,
                                     'fee_amount': 0.0003383728327220064, 'fee_token': 'USDC',
                                     'quote_quantity': 135.2698052, 'base_fee': 0.0003383728327220064,
                                     'real_price': 2933.6327304272395}))

        # profit, buy, price capped
        self.bp.simulation = True
        self.bp.fills.add_fill(Fill(new_last_fill_4))
        last = self.bp.fills.last.fill.copy()
        self.bp.last_weighted_spot = {"ETH-USDC": 2780.82}
        self.bp.market.base_fee = 1 / self.bp.last_weighted_spot["ETH-USDC"]
        self.bp.market.quote_fee = 1

    def test_next_trade_stop_loss_buy_mystery(self):
        self.bp.fills.add_fill(Fill(new_last_fill_3))
        self.bp.last_weighted_spot = {"ETH-USDC": 2601.70466}
        self.bp.market.base_fee = 1 / self.bp.last_weighted_spot["ETH-USDC"]
        self.bp.market.quote_fee = 1
        last = self.bp.fills.last.fill.copy()
        new_quote = asyncio.run(self.bp.position_close_order(for_stop_loss=True,
                                                             base_balance_=1, quote_balance_=2500))
        assert new_quote.transaction["side"] == "b"
        assert new_quote.transaction["trade"] == "USDC"
        assert round(new_quote.transaction["price"], 2) == 2615.92
        assert round(new_quote.transaction["base_quantity"], 6) == 221.731936  # 0.085143
        assert new_quote.transaction["base_quantity"] == (last["base_quantity"] - self.bp.market.base_fee) * \
               last["price"]
        assert round((last["price"] - new_quote.transaction["price"]) / last["price"], 5) >= self.bp.loss_margin

    def test_next_trade_stop_loss_buy_mystery_sim(self):
        self.bp.last_price = {"ETH-USDC": 2955.32}
        self.bp.last_weighted_spot = {"ETH-USDC": 2955.32}
        self.bp.market.base_fee = 1 / self.bp.last_weighted_spot["ETH-USDC"]
        self.bp.market.quote_fee = 1
        self.bp.fills.add_fill(Fill({'chain_id': 1000, 'id': 19, 'market': 'ETH-USDC', 'side': 's',
                                     'price': 2955.32, 'base_quantity': 0.04611, 'fill_status': 'f',
                                     'tx_hash': 'cookies', 'taker_user_id': '123', "maker_user_id": self.zk_id,
                                     'fee_amount': 0.0003383728327220064, 'fee_token': 'USDC',
                                     'quote_quantity': 135.2698052, 'base_fee': 0.0003383728327220064,
                                     'real_price': 2933.6327304272395}))

        self.bp.simulation = True
        self.bp.fills.add_fill(Fill(new_last_fill_3))
        self.bp.last_weighted_spot = {"ETH-USDC": 2601.70466}
        self.bp.market.base_fee = 1 / self.bp.last_weighted_spot["ETH-USDC"]
        self.bp.market.quote_fee = 1
        new_quote = asyncio.run(self.bp.position_close_order(for_stop_loss=True,
                                                             base_balance_=1, quote_balance_=2500))
        assert new_quote.transaction["side"] == "b"
        assert new_quote.transaction["trade"] == "USDC"
        assert round(new_quote.transaction["price"], 2) == 2604.23
        assert round(new_quote.transaction["base_quantity"], 6) == 0.084759

    def test_convert_candle_to_fill(self):
        self.bp.market.base_fee = 0.00011979
        new_fill = self.bp.convert_candle_to_fill(pd.Series(self.candle_1),
                                                  self.bp.fills.last.fill)
        assert new_fill.fill == {'market_id': 'ETH-USDC', 'traded_at': Timestamp('2021-02-03 18:35:00'),
                                 'price_low': 1640.95, 'price_high': 1644.02, 'price_open': 1641.14,
                                 'price_close': 1643.68, 'volume': 7.9705567, 'traded_on': '21-02-03', 'pm': 0.005,
                                 'sl': -0.005, 'price': 1643.68, 'market': 'ETH-USDC', 'side': 's',
                                 'fee_amount': 0.00011979, 'fee_token': 'ETH', 'base_quantity': 0.25,
                                 'quote_quantity': 410.7231035728, 'base_fee': 0.00011979,
                                 'real_price': 1642.8924142912, "fill_status": "f"}

        self.bp.fills.last.fill["side"] = "s"
        self.bp.fills.last.fill["price"] = 1610
        self.bp.market.quote_fee = 0.293
        new_fill = self.bp.convert_candle_to_fill(pd.Series(self.candle_1), self.bp.fills.last.fill)
        assert new_fill.fill == {'base_fee': 0.293,
                                 'base_quantity': 0.24884816387618025,
                                 'fee_amount': 0.293,
                                 'fee_token': 'USDC',
                                 'market': 'ETH-USDC',
                                 'market_id': 'ETH-USDC',
                                 'pm': 0.005,
                                 "fill_status": "f",
                                 'price': 1643.68,
                                 'price_close': 1643.68,
                                 'price_high': 1644.02,
                                 'price_low': 1640.95,
                                 'price_open': 1641.14,
                                 'quote_quantity': 409.02675,
                                 'real_price': 1644.8582688363758,
                                 'side': 'b',
                                 'sl': -0.005,
                                 'traded_on': '21-02-03',
                                 'volume': 7.9705567}

    def test_calculate_profit(self):
        old_fill = Fill({'chain_id': 1, 'id': 2353714, 'market': 'ETH-USDC', 'side': 'b', 'price': 1118.32,
                         'base_quantity': 0.3145751133947555, 'fill_status': 'f',
                         'tx_hash': '857e0b33196838d7462ddfb801d98e757bb68f2e0c967f762cc9996d84406eed',
                         'taker_user_id': '944285', 'maker_user_id': '1118798', 'fee_amount': 0.28140000000000004,
                         'fee_token': 'USDC', 'quote_quantity': 352.07704081162296, 'base_fee': 0.00025162744116174264,
                         'real_price': 1119.2145399302674})
        new_fill = Fill({'chain_id': 1, 'id': 2354849, 'market': 'ETH-USDC', 'side': 's', 'price': 1126.51,
                         'base_quantity': 0.31453811, 'fill_status': 'f',
                         'tx_hash': '3856ac425863b73277181cb80253693de9419aa065d66f1ef9d91815502d6cd2',
                         'taker_user_id': '944285', 'maker_user_id': '1032323', 'fee_amount': 0.00014112,
                         'fee_token': 'ETH', 'quote_quantity': 354.17135320489996, 'base_fee': 0.00014112,
                         'real_price': 1126.0045824173737})
        profit = asyncio.run(self.bp.calculate_profit(new_fill, old_fill))
        assert profit == {'ETH': 0.0, 'USDC': 2.0876793932769937, 'market': 'ETH-USDC', 'pct_return': 0.59}

    def test_update_current_profit(self):
        self.bp.simulation = False
        self.bp.stop_win_threshold = 0.0050
        self.bp.stop_win_curr_margin_pct = 0.75
        self.bp.stop_win_loss_margin_pct = 0.50
        self.bp.market.quote_fee = 0.293
        self.bp.market.base_fee = 0.00011979
        self.bp.fills.add_fill(Fill({'market': 'ETH-USDC', 'side': 'b', 'trade': 'USDC', 'price': 1634.94,
                                     'when': 'above', 'base_quantity': 0.25, 'type': 'profit', 'fee_amount': 0.293,
                                     'fee_token': 'USDC', 'quote_quantity': 409.02675, "id": 1234,
                                     'base_fee': 0.00017921201760314628, 'real_price': 1636.107}))
        asyncio.run(self.bp.update_current_profit(pd.Series(self.candle_1)))
        assert round(self.bp.current_profit_margin, 6) == 0.0034
        assert round(self.bp.current_margin_gain, 6) == 0.0034
        assert np.isnan(self.bp.stop_win)

        self.bp.fills.add_fill(Fill({'market': 'ETH-USDC', 'side': 's', 'trade': 'USDC', 'price': 1638.25,
                                     'when': 'above', 'base_quantity': 0.25, 'type': 'profit',
                                     'fee_amount': 0.00034210607340912124, 'fee_token': 'ETH',
                                     'quote_quantity': 409.02675, "id": 1235, 'base_fee': 0.00017921201760314628,
                                     'real_price': 1675}))
        asyncio.run(self.bp.update_current_profit(pd.Series(self.candle_1)))
        assert self.bp.current_profit_margin == -0.0051
        assert round(self.bp.current_margin_gain, 6) == 0.0034
        assert np.isnan(self.bp.stop_win)

        self.bp.fills.add_fill(Fill({'market': 'ETH-USDC', 'side': 's', 'trade': 'USDC', 'price': 1677.8,
                                     'when': 'above', 'base_quantity': 0.25, 'type': 'profit',
                                     'fee_amount': 0.00034210607340912124, 'fee_token': 'ETH', "id": 1236,
                                     'quote_quantity': 418.8828162650602, 'base_fee': 0.00017921201760314628,
                                     'real_price': 1700}))
        asyncio.run(self.bp.update_current_profit(pd.Series(self.candle_1)))
        assert self.bp.current_profit_margin == 0.0189
        assert round(self.bp.current_margin_gain, 6) == 0.0189
        assert self.bp.stop_win == 0.014175

        self.bp.fills.add_fill(Fill({'market': 'ETH-USDC', 'side': 's', 'trade': 'USDC', 'price': 1751.9,
                                     'when': 'above', 'base_quantity': 0.25, 'type': 'profit',
                                     'fee_amount': 0.00034210607340912124, 'fee_token': 'ETH', "id": 1237,
                                     'quote_quantity': 437.36294051204817, 'base_fee': 0.00017921201760314628,
                                     'real_price': 1760}))

        asyncio.run(self.bp.update_current_profit(pd.Series(self.candle_1)))
        assert self.bp.current_profit_margin == 0.0639
        assert round(self.bp.current_margin_gain, 6) == 0.0639
        assert self.bp.stop_win == 0.047924999999999995

    def test_process_volatility(self):
        self.bp.current_margin_gain = 0.0036
        self.bp.base_profit_margin = 0.0020
        self.bp.base_loss_margin = -0.010
        asyncio.run(self.bp.process_volatility(self.market))
        assert self.bp.profit_margin == 0.0056
        assert self.bp.loss_margin == -0.0064
        assert np.isnan(self.bp.stop_win)

        self.bp.current_margin_gain = 0.0105
        self.bp.stop_win = 0.007875
        asyncio.run(self.bp.process_volatility(self.market))
        assert self.bp.profit_margin == 0.0125
        assert self.bp.loss_margin == 0.007875

    def test_process_volatility_sim(self):

        self.bp.simulation = True
        self.bp.stop_win = np.nan
        self.bp.current_margin_gain = 0.0036
        self.bp.stop_win_threshold = 0.01
        self.bp.stop_win_loss_margin_pct = 0.75
        self.bp.stop_win_curr_margin_pct = 1.00
        candle = pd.Series({"sl": -0.0100, "pm": 0.0020, "sw_thresh": 0.01,
                            "sw_loss_margin_pct": 0.75, "sw_curr_margin_pct": 1.00})
        asyncio.run(self.bp.process_volatility(candle=candle))
        assert round(self.bp.profit_margin, 6) == 0.0136
        assert self.bp.loss_margin == -0.0064
        assert self.bp.stop_win_threshold == 0.01
        assert self.bp.stop_win_loss_margin_pct == 0.75
        assert self.bp.stop_win_curr_margin_pct == 1.00

        self.bp.stop_win = 0.005
        self.bp.current_margin_gain = 0.006
        asyncio.run(self.bp.process_volatility(candle=candle))
        assert self.bp.profit_margin == 0.016
        assert self.bp.loss_margin == 0.005
        assert self.bp.stop_win_threshold == 0.01
        assert self.bp.stop_win_loss_margin_pct == 0.75
        assert self.bp.stop_win_curr_margin_pct == 1.00

    def test_is_trade_active(self):
        self.bp.current_margin_gain = 0
        self.bp.simulation = True
        wallet = SimulationWallet(base_balance=1, quote_balance=0.5)
        self.bp.fills.add_fill(Fill({"market": "ETH-USDC",
                                     "side": "b",
                                     "trade": "ETH",
                                     "price": 2390,
                                     "when": "below",
                                     "base_quantity": 0.2,
                                     "type": "position close",
                                     "id": 3000,
                                     "fee_amount": 0.72,
                                     "fee_token": "USDC"}))
        self.bp.orders.set_by_orders_info({'base_fee': 0.00011978999999999999,
                                           'base_quantity': 0.2,
                                           'base_quantity_no_fee': 0.2,
                                           'base_quantity_with_fee': 0.2,
                                           'expected_profit_base': -0.0016,
                                           'expected_profit_quote': -3.829287982506601,
                                           'market': 'ETH-USDC',
                                           'price': 2391.8112,
                                           'price_movement': -19.28879999999981,
                                           'price_movement_no_fee': -19.28879999999981,
                                           'price_movement_with_fee': -16.36068181874998,
                                           'price_no_fee': 2391.8112,
                                           'price_with_fee': 2394.73931818125,
                                           'profit_target': -0.008,
                                           'quantile': 1.0,
                                           'quote_quantity': 478.6609978133251,
                                           'real_price': 2393.3049890666252,
                                           'side': 's',
                                           'trade': 'ETH',
                                           'type': 'stop_loss',
                                           'when': 'below'})
        profit_order_2_ = {'base_quantity': 0.2,
                           'fee_amount': 0.0003,
                           'fee_token': 'ETH',
                           'market': 'ETH-USDC',
                           'price': 2417,
                           'side': 's',
                           'trade': 'ETH',
                           'type': 'position close',
                           'when': 'below'}
        self.bp.orders.set_orders(PositionOrder({"market": "ETH-USDC",
                                                 "side": "s",
                                                 "trade": "ETH",
                                                 "price": 2417,
                                                 "when": "below",
                                                 "base_quantity": 0.2,
                                                 "type": "position close",
                                                 "fee_amount": 0.0003,
                                                 "fee_token": "ETH"}))
        candles = df_sub_select.copy()
        self.bp.profit_margin = 0.004
        self.bp.loss_margin = -0.0002
        self.bp.base_profit_margin = 0.004
        self.bp.base_loss_margin = -0.0002
        candles["sw_thresh"] = 0.01
        candles["sw_loss_margin_pct"] = 0.75
        candles["sw_curr_margin_pct"] = 1.00

        trades_activated = asyncio.run(self.bp.is_trade_active(candles.iloc[89], wallet))
        assert [self.bp.loss_margin, self.bp.profit_margin] == \
               [-0.0002, 0.0040]
        assert trades_activated.transaction == {'base_quantity': 0.2,
                                                'fee_amount': 0.0003,
                                                'fee_token': 'ETH',
                                                'market': 'ETH-USDC',
                                                'price': 2417,
                                                'side': 's',
                                                'trade': 'ETH',
                                                'type': 'position close',
                                                'when': 'below'}

        self.bp.fills.add_fill(Fill({"market": "ETH-USDC",
                                     "side": "b",
                                     "trade": "ETH",
                                     "id": 2222,
                                     "price": 2410,
                                     "when": "above",
                                     "base_quantity": 0.20,
                                     "type": "profit",
                                     "fee_amount": 0.22,
                                     "fee_token": "USDC"}))
        trades_activated = asyncio.run(self.bp.is_trade_active(candles.iloc[90], wallet))
        assert [self.bp.loss_margin, self.bp.profit_margin] == \
               [-0.0002, 0.0040]
        assert trades_activated.transaction == profit_order_2_

        trades_activated = asyncio.run(self.bp.is_trade_active(candles.iloc[91], wallet))
        assert [self.bp.loss_margin, self.bp.profit_margin] == \
               [-0.0002, 0.0040]
        assert trades_activated.transaction == profit_order_2_

        asyncio.run(self.bp.is_trade_active(candles.iloc[92], wallet))
        assert [self.bp.loss_margin, self.bp.profit_margin] == \
               [-0.0002, 0.0040]
        asyncio.run(self.bp.is_trade_active(candles.iloc[93], wallet))
        assert [self.bp.loss_margin, self.bp.profit_margin] == \
               [-0.0002, 0.0040]
        trades_activated = asyncio.run(self.bp.is_trade_active(candles.iloc[94], wallet))
        assert [self.bp.loss_margin, self.bp.profit_margin] == \
               [-0.0002, 0.0040]
        
    def tearDown(self):
        print("tearDown")

    @classmethod
    def tearDownClass(cls):
        print("tearDownClass")


if __name__ == '__main__':
    unittest.main()
