from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from freqtrade.strategy import DecimalParameter, IntParameter, CategoricalParameter
from freqtrade.persistence import Trade
from typing import Dict, List, Optional

class CryptoScalpingStrategyV0(IStrategy):
    """
    Freqtrade strategy implementation of the CryptoScalpingBotV0 logic.
    
    This strategy implements a multi-timeframe analysis approach with:
    - EMA crosses across timeframes
    - RSI conditions
    - MACD confirmation
    - Volume analysis
    - Position doubling on re-validation
    """
    # Strategy interface version
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.015,  # 1.5% profit
        "60": 0.01,  # 1% profit after 60 minutes
        "120": 0.005,  # 0.5% profit after 120 minutes
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.03  # -3% stoploss (will be overridden per trade)

    # Optimal timeframe for the strategy
    timeframe = '1m'

    # Strategy parameters (can be optimized with hyperopt)
    risk_percent = 3.0
    trailing_stop = True
    trailing_stop_positive = 0.005  # 0.5%
    trailing_stop_positive_offset = 0.02  # 2%
    trailing_only_offset_is_reached = True
    
    # Indicator parameters
    ema_short = IntParameter(5, 15, default=9, space="buy")
    ema_medium = IntParameter(15, 30, default=21, space="buy")
    ema_long = IntParameter(30, 60, default=50, space="buy")
    ema_very_long = IntParameter(80, 120, default=100, space="buy")
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    rsi_oversold = IntParameter(20, 40, default=30, space="buy")
    rsi_overbought = IntParameter(60, 80, default=70, space="buy")
    macd_fast = IntParameter(8, 15, default=12, space="buy")
    macd_slow = IntParameter(20, 30, default=26, space="buy")
    macd_signal = IntParameter(5, 10, default=9, space="buy")
    position_increase_pct = IntParameter(50, 150, default=100, space="buy")
    trailing_stop_pct = IntParameter(30, 70, default=50, space="buy")
    
    # Constructor
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Dictionary to track consecutive signals
        self.consecutive_signals = {}
        
    def informative_pairs(self):
        """
        Define additional informative pair/interval combinations to be cached
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        
        # Add multiple timeframes for each pair
        for pair in pairs:
            informative_pairs.append((pair, '3m'))
            informative_pairs.append((pair, '15m'))
            informative_pairs.append((pair, '1h'))
            informative_pairs.append((pair, '4h'))
            informative_pairs.append((pair, '1d'))
            
        return informative_pairs
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators for primary timeframe (1m)
        """
        # Initialize consecutive signals tracker if this pair doesn't exist
        pair = metadata['pair']
        if pair not in self.consecutive_signals:
            self.consecutive_signals[pair] = {
                'count': 0, 
                'direction': None
            }
        
        # Calculate EMAs
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=self.ema_short.value)
        dataframe['ema_medium'] = ta.EMA(dataframe, timeperiod=self.ema_medium.value)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=self.ema_long.value)
        dataframe['ema_very_long'] = ta.EMA(dataframe, timeperiod=self.ema_very_long.value)

        # Calculate RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # Calculate MACD
        macd = ta.MACD(
            dataframe,
            fastperiod=self.macd_fast.value,
            slowperiod=self.macd_slow.value,
            signalperiod=self.macd_signal.value
        )
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Calculate volume indicators
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']

        # Mark swing highs and lows
        dataframe['is_swing_high'] = (
            (dataframe['high'] > dataframe['high'].shift(1)) & 
            (dataframe['high'] > dataframe['high'].shift(-1))
        )
        dataframe['is_swing_low'] = (
            (dataframe['low'] < dataframe['low'].shift(1)) & 
            (dataframe['low'] < dataframe['low'].shift(-1))
        )
        
        # Get informative indicators from higher timeframes
        self.dp.add_informative_indicators(dataframe, metadata)
        
        return dataframe

    def get_informative_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add informative indicators for higher timeframes
        """
        # For 3m timeframe
        dataframe_3m = self.dp.get_pair_dataframe(
            pair=metadata['pair'],
            timeframe='3m'
        )
        dataframe_3m['ema_short'] = ta.EMA(dataframe_3m, timeperiod=self.ema_short.value)
        dataframe_3m['ema_medium'] = ta.EMA(dataframe_3m, timeperiod=self.ema_medium.value)
        dataframe_3m['rsi'] = ta.RSI(dataframe_3m, timeperiod=self.rsi_period.value)
        macd_3m = ta.MACD(
            dataframe_3m,
            fastperiod=self.macd_fast.value,
            slowperiod=self.macd_slow.value,
            signalperiod=self.macd_signal.value
        )
        dataframe_3m['macdhist'] = macd_3m['macdhist']
        
        # For 15m timeframe
        dataframe_15m = self.dp.get_pair_dataframe(
            pair=metadata['pair'],
            timeframe='15m'
        )
        dataframe_15m['ema_short'] = ta.EMA(dataframe_15m, timeperiod=self.ema_short.value)
        dataframe_15m['ema_medium'] = ta.EMA(dataframe_15m, timeperiod=self.ema_medium.value)
        dataframe_15m['rsi'] = ta.RSI(dataframe_15m, timeperiod=self.rsi_period.value)
        
        # For 1h timeframe (if available)
        try:
            dataframe_1h = self.dp.get_pair_dataframe(
                pair=metadata['pair'],
                timeframe='1h'
            )
            dataframe_1h['ema_short'] = ta.EMA(dataframe_1h, timeperiod=self.ema_short.value)
            dataframe_1h['ema_medium'] = ta.EMA(dataframe_1h, timeperiod=self.ema_medium.value)
        except:
            dataframe_1h = None
            
        # For 4h timeframe (if available)
        try:
            dataframe_4h = self.dp.get_pair_dataframe(
                pair=metadata['pair'],
                timeframe='4h'
            )
            dataframe_4h['ema_short'] = ta.EMA(dataframe_4h, timeperiod=self.ema_short.value)
            dataframe_4h['ema_medium'] = ta.EMA(dataframe_4h, timeperiod=self.ema_medium.value)
        except:
            dataframe_4h = None
            
        # For 1d timeframe (if available)
        try:
            dataframe_1d = self.dp.get_pair_dataframe(
                pair=metadata['pair'],
                timeframe='1d'
            )
            dataframe_1d['ema_short'] = ta.EMA(dataframe_1d, timeperiod=self.ema_short.value)
            dataframe_1d['ema_medium'] = ta.EMA(dataframe_1d, timeperiod=self.ema_medium.value)
        except:
            dataframe_1d = None
            
        # Store timeframe data for use in populate_buy_trend and populate_sell_trend
        self.dataframe_3m = dataframe_3m
        self.dataframe_15m = dataframe_15m
        self.dataframe_1h = dataframe_1h
        self.dataframe_4h = dataframe_4h
        self.dataframe_1d = dataframe_1d
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populate the buy trend column
        """
        pair = metadata['pair']
        
        # Initialize signals
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Get the latest values from higher timeframes by resampling
        # 1m -> 3m
        df_3m = self.dp.resample(dataframe, '3m')
        # 1m -> 15m
        df_15m = self.dp.resample(dataframe, '15m')
        # 1m -> 1h (Optional)
        df_1h = self.dp.resample(dataframe, '1h')
        # 1m -> 4h (Optional)
        df_4h = self.dp.resample(dataframe, '4h')
        # 1m -> 1d (Optional)
        df_1d = self.dp.resample(dataframe, '1d')
        
        # Buy signal for LONG entries - Alignment of EMAs across timeframes
        dataframe.loc[
            (
                # 1m conditions
                (dataframe['close'] > dataframe['ema_medium']) &  
                (dataframe['ema_medium'] > dataframe['ema_long']) &
                
                # 3m conditions 
                (df_3m['close'] > df_3m['ema_medium']) &
                
                # 15m conditions
                (df_15m['ema_short'] > df_15m['ema_medium']) &
                
                # Optional higher timeframe conditions if available
                (df_1h['ema_short'] > df_1h['ema_medium'] if df_1h is not None else True) &
                (df_4h['ema_short'] > df_4h['ema_medium'] if df_4h is not None else True) &
                (df_1d['ema_short'] > df_1d['ema_medium'] if df_1d is not None else True) &
                
                # RSI conditions - not oversold on higher timeframes
                (dataframe['rsi'] > 40) & (dataframe['rsi'] < 70) &
                (df_3m['rsi'] > 40) &
                (df_15m['rsi'] > 40) &
                
                # MACD conditions
                (dataframe['macdhist'] > 0) &
                (dataframe['macd'] > dataframe['macdsignal']) &
                (df_3m['macdhist'] > 0) &
                
                # Volume confirmation
                (dataframe['volume_ratio'] > 1.2)
            ),
            'enter_long'] = 1
            
        # Sell signal for SHORT entries - Opposite alignment of EMAs across timeframes
        dataframe.loc[
            (
                # 1m conditions
                (dataframe['close'] < dataframe['ema_medium']) &  
                (dataframe['ema_medium'] < dataframe['ema_long']) &
                
                # 3m conditions 
                (df_3m['close'] < df_3m['ema_medium']) &
                
                # 15m conditions
                (df_15m['ema_short'] < df_15m['ema_medium']) &
                
                # Optional higher timeframe conditions if available
                (df_1h['ema_short'] < df_1h['ema_medium'] if df_1h is not None else True) &
                (df_4h['ema_short'] < df_4h['ema_medium'] if df_4h is not None else True) &
                (df_1d['ema_short'] < df_1d['ema_medium'] if df_1d is not None else True) &
                
                # RSI conditions - not overbought on higher timeframes
                (dataframe['rsi'] < 60) & (dataframe['rsi'] > 30) &
                (df_3m['rsi'] < 60) &
                (df_15m['rsi'] < 60) &
                
                # MACD conditions
                (dataframe['macdhist'] < 0) &
                (dataframe['macd'] < dataframe['macdsignal']) &
                (df_3m['macdhist'] < 0) &
                
                # Volume confirmation
                (dataframe['volume_ratio'] > 1.2)
            ),
            'enter_short'] = 1
            
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populate the sell trend column
        """
        # We'll use stoploss/ROI/trailing stop for exits
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Add technical invalidation (EMA crossovers) for exiting positions
        dataframe.loc[
            (
                (dataframe['ema_short'].shift(1) > dataframe['ema_medium'].shift(1)) &
                (dataframe['ema_short'] < dataframe['ema_medium'])
            ),
            'exit_long'] = 1
            
        dataframe.loc[
            (
                (dataframe['ema_short'].shift(1) < dataframe['ema_medium'].shift(1)) &
                (dataframe['ema_short'] > dataframe['ema_medium'])
            ),
            'exit_short'] = 1
            
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           **kwargs) -> float:
        """
        Customize stake amount based on consecutive signals logic
        """
        # Get pair's consecutive signal count
        if pair in self.consecutive_signals:
            signal_info = self.consecutive_signals[pair]
            # If we have consecutive signals in same direction, increase stake
            if signal_info['count'] > 0:
                # Increase by position_increase_pct %
                increase_factor = 1 + (self.position_increase_pct.value / 100)
                return min(proposed_stake * increase_factor, max_stake)
        
        return proposed_stake
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic, trailing stop functionality
        """
        # Get pair's signal info
        if pair in self.consecutive_signals:
            signal_info = self.consecutive_signals[pair]
            
            # Calculate trailing stop based on initial risk
            if current_profit > 0:
                # Check if this is a technical invalidation scenario
                # This is simplified since we don't have access to previous candles here
                # Full implementation would check EMA crossover
                
                # Use trailing_stop_pct for position management
                trailing_offset = self.trailing_stop_pct.value / 100
                return current_profit * trailing_offset
            
        # Default stoploss
        return self.stoploss
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, **kwargs) -> bool:
        """
        Called before placing an entry order, can be used to track consecutive signals
        """
        # Get trade direction from order_type
        is_short = order_type.startswith('sell')
        direction = 'SHORT' if is_short else 'LONG'
        
        # Update consecutive signal counter
        if pair in self.consecutive_signals:
            signal_info = self.consecutive_signals[pair]
            if signal_info['direction'] == direction:
                signal_info['count'] += 1
            else:
                signal_info['count'] = 1
                signal_info['direction'] = direction
        else:
            self.consecutive_signals[pair] = {
                'count': 1,
                'direction': direction
            }
        
        return True
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Called before placing an exit order, can be used to reset consecutive signal counter
        """
        # Reset counter unless this is a technical invalidation exit
        if exit_reason != 'exit_signal':
            if pair in self.consecutive_signals:
                self.consecutive_signals[pair] = {
                    'count': 0,
                    'direction': None
                }
        
        return True


class CryptoScalpingStrategyV1(CryptoScalpingStrategyV0):
    """
    FreqTrade implementation of CryptoScalpingBotV1
    
    Enhanced version of V0 with improved trailing stop logic and position
    doubling after same-direction invalidations.
    """
    # Class parameters
    position_increase_pct_on_revalidation = IntParameter(50, 150, default=100, space="buy")
    trailing_stop_pct = IntParameter(30, 70, default=50, space="buy")
    invalidation_r_factor = DecimalParameter(0.3, 0.7, default=0.5, space="sell")
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add V1-specific indicators on top of V0 indicators
        """
        # Get base indicators
        dataframe = super().populate_indicators(dataframe, metadata)
        
        # Additional V1 specific indicators could be added here
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           **kwargs) -> float:
        """
        V1 enhanced stake amount calculation for consecutive signals
        """
        # Get pair's consecutive signal count
        if pair in self.consecutive_signals:
            signal_info = self.consecutive_signals[pair]
            # If we have consecutive signals in same direction, increase stake by V1 factor
            if signal_info['count'] > 0:
                # Increase by position_increase_pct_on_revalidation %
                increase_factor = 1 + (self.position_increase_pct_on_revalidation.value / 100)
                return min(proposed_stake * increase_factor, max_stake)
        
        return proposed_stake
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Enhanced V1 stoploss with more sophisticated trailing stop
        """
        # For short positions, we need to invert the profit calculation
        # Freqtrade already handles this, so current_profit is correct
        
        # Calculate initial risk - we'll use a simplified version since we don't store initial stop loss
        initial_risk = abs(self.stoploss)
        
        # If we've reached trailing stop activation threshold (50% of initial risk by default)
        if current_profit >= initial_risk * (self.trailing_stop_pct.value / 100):
            # Calculate trailing stop distance using R factor
            trailing_distance = initial_risk * self.invalidation_r_factor.value
            
            # Return new stoploss level
            return current_profit - trailing_distance
            
        # Default stoploss
        return self.stoploss


class CryptoScalpingStrategyV2(CryptoScalpingStrategyV1):
    """
    FreqTrade implementation of CryptoScalpingBotV2
    
    Enhanced version of V1 with ATR-based position sizing and stoploss,
    as well as additional checks using ema_very_long.
    """
    # Additional V2 parameters
    atr_period = IntParameter(10, 20, default=14, space="buy")
    atr_stoploss_factor = DecimalParameter(1.0, 3.0, default=2.0, space="buy")
    atr_trailing_factor = DecimalParameter(1.0, 2.0, default=1.5, space="sell")
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add V2-specific indicators on top of V1 indicators
        """
        # Get base indicators
        dataframe = super().populate_indicators(dataframe, metadata)
        
        # V2 specific: ATR calculation
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        V2 enhanced entry signals with additional conditions
        """
        # Get base entry signals
        dataframe = super().populate_entry_trend(dataframe, metadata)
        
        # Add V2 specific condition: Check ema_very_long for long-term trend confirmation
        mask = (
            (dataframe['enter_long'] == 1) &
            (dataframe['close'] > dataframe['ema_very_long'])
        )
        dataframe.loc[~mask, 'enter_long'] = 0
        
        mask = (
            (dataframe['enter_short'] == 1) &
            (dataframe['close'] < dataframe['ema_very_long'])
        )
        dataframe.loc[~mask, 'enter_short'] = 0
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           **kwargs) -> float:
        """
        V2 enhanced stake amount calculation based on ATR
        """
        # Get pair's dataframe to retrieve ATR value
        dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)
        
        if dataframe is not None and not dataframe.empty and 'atr' in dataframe.columns:
            # Get latest ATR value
            latest_atr = dataframe['atr'].iloc[-1]
            
            if latest_atr > 0:
                # Calculate position size based on ATR
                risk_amount = self.config['stake_amount'] * (self.risk_percent / 100)
                atr_position_size = risk_amount / (latest_atr * self.atr_stoploss_factor.value)
                
                # Convert to stake currency
                atr_stake = atr_position_size * current_rate
                
                # Conservative position size increase for consecutive signals (half of V1 rate)
                if pair in self.consecutive_signals and self.consecutive_signals[pair]['count'] > 0:
                    increase_factor = 1 + (self.position_increase_pct_on_revalidation.value / 200)
                    atr_stake *= increase_factor
                
                # Ensure within limits
                return min(max(atr_stake, min_stake), max_stake)
        
        # Fallback to parent method if ATR calculation fails
        return super().custom_stake_amount(pair, current_time, current_rate, 
                                          proposed_stake, min_stake, max_stake, **kwargs)
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        V2 enhanced stoploss with ATR-based trailing
        """
        # Get pair's dataframe to retrieve ATR value
        dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)
        
        if dataframe is not None and not dataframe.empty and 'atr' in dataframe.columns:
            # Get latest ATR value
            latest_atr = dataframe['atr'].iloc[-1]
            
            # For profit positions, use dynamic ATR trailing
            if current_profit > 0:
                # V2 activates trailing at 1.5x initial risk
                initial_risk = abs(self.stoploss)
                
                if current_profit >= initial_risk * 1.5:
                    # Calculate trailing stop based on current ATR
                    atr_trailing_distance = latest_atr * self.atr_trailing_factor.value / current_rate
                    return current_profit - atr_trailing_distance
        
        # Fallback to parent method if ATR calculation fails
        return super().custom_stoploss(pair, trade, current_time, current_rate, current_profit, **kwargs)
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Called before placing an exit order, logs ATR at exit for analysis
        """
        # Get ATR at exit for analysis
        dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)
        
        if dataframe is not None and not dataframe.empty and 'atr' in dataframe.columns:
            atr_at_exit = dataframe['atr'].iloc[-1]
            self.log_once(f"{pair} exit with ATR: {atr_at_exit}", logger.INFO)
        
        return super().confirm_trade_exit(pair, trade, order_type, amount, rate, 
                                         time_in_force, exit_reason, current_time, **kwargs)
