# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import copy
import warnings
import numpy as np
import pandas as pd

from typing import Dict, List, Text, Tuple, Union
from abc import ABC

from qlib.data import D
from qlib.data.dataset import Dataset
from qlib.model.base import BaseModel
from qlib.strategy.base import BaseStrategy
from qlib.backtest.position import Position
from qlib.backtest.signal import Signal, create_signal_from
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.log import get_module_logger
from qlib.utils import get_pre_trading_date, load_dataset


class BaseSignalStrategy(BaseStrategy, ABC):
    def __init__(
        self,
        *,
        signal: Union[
            Signal, Tuple[BaseModel, Dataset], List, Dict, Text, pd.Series, pd.DataFrame
        ] = None,
        model=None,
        dataset=None,
        risk_degree: float = 0.95,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        signal :
            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`
            the decision of the strategy will base on the given signal
        risk_degree : float
            position percentage of total value.
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:
                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.

        """
        super().__init__(
            level_infra=level_infra,
            common_infra=common_infra,
            trade_exchange=trade_exchange,
            **kwargs,
        )

        self.risk_degree = risk_degree

        # This is trying to be compatible with previous version of qlib task config
        if model is not None and dataset is not None:
            warnings.warn(
                "`model` `dataset` is deprecated; use `signal`.", DeprecationWarning
            )
            signal = model, dataset

        self.signal: Signal = create_signal_from(signal)

    def get_risk_degree(self, trade_step=None):
        """get_risk_degree
        Return the proportion of your total value you will use in investment.
        Dynamically risk_degree will result in Market timing.
        """
        # It will use 95% amount of your total value by default
        return self.risk_degree


class TimingStrategy(BaseSignalStrategy):
    """Timing Strategy for Market Timing

    This strategy is designed for market timing based on return prediction.
    It adjusts position size based on predicted returns:
    - If predicted return > 0: Buy/Hold (high position)
    - If predicted return <= 0: Sell (low/zero position)

    Parameters
    -----------
    high_position_ratio : float
        Position ratio when predicted return is positive (default: 0.95)
    low_position_ratio : float
        Position ratio when predicted return is negative (default: 0.0)
    threshold : float
        Threshold for determining positive/negative returns (default: 0.0)
    use_benchmark_weight : bool
        Whether to use benchmark weight when position is high (default: False)
    benchmark : str
        Benchmark index name when use_benchmark_weight is True (default: "csi500")
    """

    def __init__(
        self,
        *,
        high_position_ratio: float = 0.95,
        low_position_ratio: float = 0.0,
        threshold: float = 0.0,
        use_benchmark_weight: bool = False,
        benchmark: str = "csi500",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.high_position_ratio = high_position_ratio
        self.low_position_ratio = low_position_ratio
        self.threshold = threshold
        self.use_benchmark_weight = use_benchmark_weight
        self.benchmark = benchmark
        self.logger = get_module_logger("TimingStrategy")

    def get_risk_degree(self, trade_step=None):
        """Dynamic risk degree based on predicted returns

        Returns
        -------
        float
            Position ratio based on prediction signal
        """
        # Get current trading time
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(
            trade_step, shift=1
        )

        # Get prediction signal
        pred_score = self.signal.get_signal(
            start_time=pred_start_time, end_time=pred_end_time
        )

        if pred_score is None:
            self.logger.warning(
                f"No prediction signal available for {trade_start_time}, using low position"
            )
            return self.low_position_ratio

        # Handle different signal formats
        if isinstance(pred_score, pd.DataFrame):
            # Use the first column if multiple signals
            pred_score = pred_score.iloc[:, 0]

        # Calculate average prediction (for portfolio-level timing)
        if isinstance(pred_score, pd.Series):
            avg_prediction = pred_score.mean()
        else:
            avg_prediction = pred_score

        # Determine position based on prediction
        if avg_prediction > self.threshold:
            position_ratio = self.high_position_ratio
            self.logger.info(
                f"Positive prediction ({avg_prediction:.4f}), using high position ({position_ratio})"
            )
        else:
            position_ratio = self.low_position_ratio
            self.logger.info(
                f"Negative prediction ({avg_prediction:.4f}), using low position ({position_ratio})"
            )

        return position_ratio

    def generate_trade_decision(self, execute_result=None):
        """Generate trading decision based on timing strategy

        Returns
        -------
        TradeDecisionWO
            Trading decision with buy/sell orders
        """
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(
            trade_step, shift=1
        )

        # Get prediction signal
        pred_score = self.signal.get_signal(
            start_time=pred_start_time, end_time=pred_end_time
        )

        if pred_score is None:
            return TradeDecisionWO([], self)

        # Handle different signal formats
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]

        # Calculate average prediction
        if isinstance(pred_score, pd.Series):
            avg_prediction = pred_score.mean()
        else:
            avg_prediction = pred_score

        # Determine action based on prediction
        if avg_prediction > self.threshold:
            # Positive prediction: Buy/Hold
            return self._generate_buy_decision(
                trade_start_time, trade_end_time, pred_score
            )
        else:
            # Negative prediction: Sell
            return self._generate_sell_decision(trade_start_time, trade_end_time)

    def _generate_buy_decision(self, trade_start_time, trade_end_time, pred_score):
        """Generate buy decision when prediction is positive"""
        current_temp = copy.deepcopy(self.trade_position)
        buy_order_list = []

        # Get current cash
        cash = current_temp.get_cash()
        available_cash = cash * self.get_risk_degree()

        if self.use_benchmark_weight:
            # Use benchmark weights for allocation
            target_stocks = self._get_benchmark_stocks(trade_start_time)
        else:
            # Use top stocks based on prediction score
            if isinstance(pred_score, pd.Series):
                target_stocks = (
                    pred_score.sort_values(ascending=False).head(50).index.tolist()
                )
            else:
                # If no stock-specific prediction, use benchmark
                target_stocks = self._get_benchmark_stocks(trade_start_time)

        # Generate buy orders
        if len(target_stocks) > 0:
            value_per_stock = available_cash / len(target_stocks)

            for stock_id in target_stocks:
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY,
                ):
                    continue

                # Calculate buy amount
                buy_price = self.trade_exchange.get_deal_price(
                    stock_id=stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY,
                )

                buy_amount = value_per_stock / buy_price
                factor = self.trade_exchange.get_factor(
                    stock_id=stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                )
                buy_amount = self.trade_exchange.round_amount_by_trade_unit(
                    buy_amount, factor
                )

                if buy_amount > 0:
                    buy_order = Order(
                        stock_id=stock_id,
                        amount=buy_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=Order.BUY,
                    )
                    buy_order_list.append(buy_order)

        return TradeDecisionWO(buy_order_list, self)

    def _generate_sell_decision(self, trade_start_time, trade_end_time):
        """Generate sell decision when prediction is negative"""
        current_temp = copy.deepcopy(self.trade_position)
        sell_order_list = []

        # Sell all current positions
        current_stock_list = current_temp.get_stock_list()

        for stock_id in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=stock_id,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.SELL,
            ):
                continue

            # Get current stock amount
            sell_amount = current_temp.get_stock_amount(code=stock_id)

            if sell_amount > 0:
                sell_order = Order(
                    stock_id=stock_id,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,
                )
                sell_order_list.append(sell_order)

        return TradeDecisionWO(sell_order_list, self)

    def _get_benchmark_stocks(self, trade_date):
        """Get benchmark stocks for allocation"""
        try:
            # Get benchmark weights
            bench_weight = D.features(
                D.instruments("all"),
                [f"${self.benchmark}_weight"],
                start_time=trade_date,
                end_time=trade_date,
            ).squeeze()

            if isinstance(bench_weight, pd.Series):
                bench_weight.index = bench_weight.index.droplevel(level="datetime")
                # Return stocks with positive weights
                return bench_weight[bench_weight > 0].index.tolist()
            else:
                return []
        except Exception as e:
            self.logger.warning(f"Failed to get benchmark stocks: {e}")
            return []

