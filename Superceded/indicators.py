""""""
""" indicators.py	  	   		  	  		  		  		    	 		 		   		 		  

Code implementing your indicators as functions that operate on DataFrames. There is no defined API for indicators.py, 
but when it runs, the main method should generate the charts that will illustrate your indicators in the report. 

Student Name: Jorge Salvador Aguilar Moreno  		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jmoreno62	  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845924		  	   		  	  		  		  		    	 		 		   		 		  
"""

import math
import sys
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import marketsimcode as mktsim
from util import get_data, plot_data

# Change implemented for Project 8: Indicators must return a single result vector.

class indicators(object):
    """
    This is a 'indicators' class

    :param symbol: ticker symbol to be analyzed
    :type symbol: str
    :param sd: start date
    :type sd: dt.datetime object
    :param ed: end date
    :type ed: dt.datetime object
    :param sv: start value of portfolio
    :type sv: float
    """

    def __init__(self, symbol="AAPL",
                 sd=dt.datetime(2010, 1, 1),
                 ed=dt.datetime(2011, 12, 31),
                 sv=100000):
        """
        Constructor method
        """
        self.symbol = symbol
        self.sd = sd
        self.ed = ed
        self.sv = sv

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jmoreno62"


    def SMA(self,lookback):
        # add_day = 0
        # prev_days = 0
        price = get_data([self.symbol], pd.date_range(self.sd, self.ed), addSPY=True)

        if lookback > 0:
            prev_days = max(5, 2 * lookback)     # Consider lookback of double the defined lookback in function
            ext_price = get_data([self.symbol], pd.date_range(self.sd - dt.timedelta(days=prev_days), self.ed),
                                 addSPY=True)   # Get extended price dataframe
            ext_price = ext_price.iloc[-(price.shape[0]+lookback):]
        else:
            ext_price = price.copy()
        # # Extended price 'ext_price' DataFrame to account for trading days before start day equal to lookback
        # while prev_days < lookback:
        #     ext_price = get_data([self.symbol], pd.date_range(self.sd - dt.timedelta(days=lookback+add_day), self.ed), addSPY=True)
        #     add_day += 1
        #     prev_days = len(ext_price) - len(price)
        # del add_day,prev_days

        if 'SPY' not in self.symbol:  # Once we have the trading days, drop SPY if it is not in portfolio
            price = price.drop(columns=['SPY'])
            ext_price = ext_price.drop(columns=['SPY'])
        ext_price.fillna(method="ffill", inplace=True)  # fill forward first
        ext_price.fillna(method="bfill", inplace=True)  # fill backwards second

        ### CALCULATE SMA FOR THE ENTIRE DATE RANGE FOR ALL THE SYMBOLS
        sma = ext_price.cumsum()  # Vectorized!
        sma.values[lookback:, :] = (sma.values[lookback:, :] - sma.values[:-lookback, :]) / lookback  # Vectorized!
        sma.ix[:lookback, :] = np.nan  # Vectorized!
        sma = sma.ix[lookback:,:]    #   Ignore previous days
        return sma

        # Alternative:
        # sma = ext_price.rolling(window=lookback,min_periods=lookback).mean()
        # sma = pd.rolling_mean(ext_price,window=lookback,min_periods=lookback)
        # return sma


    def EMA(self,lookback):

        sma = self.SMA(lookback)    # Call SMA with the same lookback window
        # For derivation, refer to https://school.stockcharts.com/doku.php?id=technical_indicators:moving_averages
        multiplier = 2 / (lookback + 1) # Refer to
        price = get_data([self.symbol], pd.date_range(self.sd, self.ed), addSPY=True)

        # CALCULATE EMA
        ema = pd.DataFrame(index=sma.index,columns=sma.columns)
        ema.iloc[0] = sma.iloc[0]   # Start value of EMA is SMA

        for i in range(1,len(sma)):
            ema.iloc[i] = (price.iloc[i] - ema.iloc[i-1]) * multiplier + ema.iloc[i-1]

        return ema


    def PercentB(self, lookback):

        sma = self.SMA(lookback)

        add_day = 0
        prev_days = 0
        price = get_data([self.symbol], pd.date_range(self.sd, self.ed), addSPY=True)

        # Extended price 'ext_price' DataFrame to account for trading days before start day equal to lookback
        while prev_days < lookback:
            ext_price = get_data([self.symbol], pd.date_range(self.sd - dt.timedelta(days=lookback+add_day), self.ed), addSPY=True)
            add_day += 1
            prev_days = len(ext_price) - len(price)
        del add_day,prev_days

        if 'SPY' not in self.symbol:  # Once we have the trading days, drop SPY if it is not in portfolio
            price = price.drop(columns=['SPY'])
            ext_price = ext_price.drop(columns=['SPY'])
        ext_price.fillna(method="ffill", inplace=True)  # fill forward first
        ext_price.fillna(method="bfill", inplace=True)  # fill backwards second


        ### CALCULATE BOLLINGER BANDS (14 DAY) OVER THE ENTIRE PERIOD, THEN BB%
        rolling_std = ext_price.rolling(window=lookback, min_periods=lookback).std()
        top_band = sma + (2 * rolling_std)
        bottom_band = sma - (2 * rolling_std)
        bbp = (ext_price - bottom_band) / (top_band - bottom_band)

        # Ignore previous days
        rolling_std = rolling_std.ix[lookback:, :]
        top_band = top_band.ix[lookback:, :]
        bottom_band = bottom_band.ix[lookback:, :]
        bbp = bbp.ix[lookback:, :]

        # # Create DataFrame to help visualize all variables in plots
        # df_bbp = pd.concat([bbp, sma, price, top_band, bottom_band], axis=1)
        # df_bbp.columns = ['bbp', 'sma', 'price', 'top_band', 'bottom_band']
        # return df_bbp

        # For project 8, indicators must return a single results vector
        bbp.columns = ['bbp']
        return bbp


    def price_SMA(self, lookback=50):

        sma = self.SMA(lookback)

        price = get_data([self.symbol], pd.date_range(self.sd, self.ed), addSPY=True)

        if 'SPY' not in self.symbol:  # Once we have the trading days, drop SPY if it is not in portfolio
            price = price.drop(columns=['SPY'])

        # # Create DataFrame to help visualize all variables in plots
        # df_price_SMA = pd.concat([price/sma, price, sma], axis=1)
        # df_price_SMA.columns = ['price/sma', 'price', 'sma']
        # return df_price_SMA

        # For project 8, indicators must return a single results vector
        price_SMA = price/sma
        price_SMA.columns = ['price/sma']
        return price_SMA


    def price_EMA(self, lookback=50):

        # sma = self.SMA(lookback)
        ema = self.EMA(lookback)

        price = get_data([self.symbol], pd.date_range(self.sd, self.ed), addSPY=True)

        if 'SPY' not in self.symbol:  # Once we have the trading days, drop SPY if it is not in portfolio
            price = price.drop(columns=['SPY'])

        # # Create DataFrame to help visualize all variables in plots
        # df_price_EMA = pd.concat([price/ema, price, ema, sma], axis=1)
        # df_price_EMA.columns = ['price/ema', 'price', 'ema', 'sma']
        # return df_price_EMA

        # For project 8, indicators must return a single results vector
        price_EMA = price/ema
        price_EMA.columns = ['price/ema']
        return price_EMA


    def RSI(self,lookback):

        add_day = 0
        prev_days = 0
        price = get_data([self.symbol], pd.date_range(self.sd, self.ed), addSPY=True)

        # Extended price 'ext_price' DataFrame to account for trading days before start day equal to lookback
        while prev_days < lookback:
            ext_price = get_data([self.symbol], pd.date_range(self.sd - dt.timedelta(days=lookback+add_day), self.ed), addSPY=True)
            add_day += 1
            prev_days = len(ext_price) - len(price)
        del add_day,prev_days

        if 'SPY' not in self.symbol:  # Once we have the trading days, drop SPY if it is not in portfolio
            price = price.drop(columns=['SPY'])
            ext_price = ext_price.drop(columns=['SPY'])
        price.fillna(method="ffill", inplace=True)  # fill forward first
        price.fillna(method="bfill", inplace=True)  # fill backwards second
        ext_price.fillna(method="ffill", inplace=True)  # fill forward first
        ext_price.fillna(method="bfill", inplace=True)  # fill backwards second


        ### Calculate daily returns for 'delta'
        daily_rets = ext_price.copy()
        daily_rets.values[1:, :] = ext_price.values[1:, :] - ext_price.values[:-1, :]
        daily_rets.values[0, :] = np.nan


        ### CALCULATE RELATIVE STRENGTH, THEN RSI
        up_rets = daily_rets[daily_rets  >=0].fillna(0).cumsum()
        down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()

        rsi = ext_price.copy()  # For RSI
        rsi.ix[:,:] = 0

        up_gain = ext_price.copy()
        up_gain.ix[:,:] = 0
        up_gain.values[lookback:,:] = up_rets.values[lookback:,:] - up_rets.values[:-lookback,:]

        down_loss = ext_price.copy()
        down_loss.ix[:,:] = 0
        down_loss.values[lookback:,:] = down_rets.values[lookback:,:] - down_rets.values[:-lookback,:]

        # Calculate the RS and RSI all at once
        avg_gain = up_gain / lookback
        avg_loss = down_loss / lookback
        rs = (avg_gain) / (avg_loss)
        rsi = 100 - (100 / (1 + rs))
        rsi.ix[:lookback,:] = np.nan

        # Inf results mean down_loss was 0. Those should be RSI 100
        rsi[rsi == np.inf] = 100

        # Ignore previous days
        rsi = rsi.ix[lookback:,:]
        rs = rs.ix[lookback:, :]
        avg_gain = avg_gain.ix[lookback:, :]
        avg_loss = avg_loss.ix[lookback:, :]
        daily_rets = daily_rets.ix[lookback:, :]
        up_rets = up_rets.ix[lookback:, :]
        down_rets = down_rets.ix[lookback:, :]

        # # Create DataFrame to help visualize all variables in plots
        # df_rsi = pd.concat([rsi, rs, price, avg_gain, avg_loss, daily_rets, up_rets, down_rets], axis=1)
        # df_rsi.columns = ['rsi', 'rs', 'price', 'avg_gain', 'avg_loss', 'daily_rets', 'up_rets', 'down_rets']
        # return df_rsi

        # For project 8, indicators must return a single results vector
        rsi.columns = ['rsi']
        return rsi


    def ROC(self,lookback = 12):

        sma = self.SMA(lookback)    # Call SMA

        add_day = 0
        prev_days = 0
        price = get_data([self.symbol], pd.date_range(self.sd, self.ed), addSPY=True)

        # Extended price 'ext_price' DataFrame to account for trading days before start day equal to lookback
        while prev_days < lookback:
            ext_price = get_data([self.symbol], pd.date_range(self.sd - dt.timedelta(days=lookback+add_day), self.ed), addSPY=True)
            add_day += 1
            prev_days = len(ext_price) - len(price)
        del add_day,prev_days

        if 'SPY' not in self.symbol:  # Once we have the trading days, drop SPY if it is not in portfolio
            price = price.drop(columns=['SPY'])
            ext_price = ext_price.drop(columns=['SPY'])
        ext_price.fillna(method="ffill", inplace=True)  # fill forward first
        ext_price.fillna(method="bfill", inplace=True)  # fill backwards second

        ### CALCULATE ROC
        roc = ( (ext_price - ext_price.shift(lookback)) / ext_price.shift(lookback)) * 100
        roc = roc.ix[lookback:,:]    #   Ignore previous days

        # # Create DataFrame to help visualize all variables in plots
        # df_ROC = pd.concat([roc, price, sma], axis=1)
        # df_ROC.columns = ['roc', 'price', 'sma']
        # return df_ROC

        # For project 8, indicators must return a single results vector
        roc.columns = ['roc']
        return roc


    def FSO(self,lookback = 14):

        sma = self.SMA(lookback)    # Call SMA

        add_day = 0
        prev_days = 0
        price = get_data([self.symbol], pd.date_range(self.sd, self.ed), addSPY=True, colname="Adj Close")
        # xprice = get_data([self.symbol], pd.date_range(self.sd, self.ed), addSPY=True, colname="Close")
        # high = get_data([self.symbol], pd.date_range(self.sd, self.ed), addSPY=True, colname="High")
        # low = get_data([self.symbol], pd.date_range(self.sd, self.ed), addSPY=True, colname="Low")

        # Extended price 'ext_price' DataFrame to account for trading days before start day equal to lookback
        while prev_days < lookback:
            ext_price = get_data([self.symbol], pd.date_range(self.sd - dt.timedelta(days=lookback+add_day), self.ed), addSPY=True, colname="Adj Close")
            ext_xprice = get_data([self.symbol], pd.date_range(self.sd - dt.timedelta(days=lookback+add_day), self.ed), addSPY=True, colname="Close")
            ext_high = get_data([self.symbol], pd.date_range(self.sd - dt.timedelta(days=lookback+add_day), self.ed), addSPY=True, colname="High")
            ext_low = get_data([self.symbol], pd.date_range(self.sd - dt.timedelta(days=lookback+add_day), self.ed), addSPY=True, colname="Low")
            add_day += 1
            prev_days = len(ext_price) - len(price)
        del add_day,prev_days

        if 'SPY' not in self.symbol:  # Once we have the trading days, drop SPY if it is not in portfolio
            price = price.drop(columns=['SPY'])
            ext_price = ext_price.drop(columns=['SPY'])
            ext_xprice = ext_xprice.drop(columns=['SPY'])
            ext_high = ext_high.drop(columns=['SPY'])
            ext_low = ext_low.drop(columns=['SPY'])
        ext_price.fillna(method="ffill", inplace=True)  # fill forward first
        ext_price.fillna(method="bfill", inplace=True)  # fill backwards second

        # Adjust High and Low using 'ext_price / ext_xprice' ratio
        Adjustment = ext_price / ext_xprice
        ext_high = ext_high * Adjustment
        ext_low = ext_low * Adjustment
        # del Adjustment, ext_xprice


        # highest_high_test = pd.DataFrame(index = ext_price.index)
        # lowest_low_test = pd.DataFrame(index=ext_price.index)

        highest_high = ext_price.copy()
        highest_high.ix[:,:] = 0

        lowest_low = ext_price.copy()
        lowest_low.ix[:,:] = 0

        ### CALCULATE FSO
        # NOTE: Vectorize for future projects
        for i in range(lookback,len(ext_high)):
            highest_high.ix[i,0] = ext_high.ix[i-lookback:i+1,0].max() # Jorge don't forget, the 'i+1' was causing a problem
            lowest_low.ix[i,0] = ext_low.ix[i-lookback:i+1,0].min()

        fso = (ext_price - lowest_low) / (highest_high - lowest_low) * 100

        # Ignore previous days
        fso = fso.ix[lookback:,:]
        highest_high = highest_high.ix[lookback:, :]
        lowest_low = lowest_low.ix[lookback:, :]

        # Calculate %D
        # Alternative:
        DP = fso.rolling(window=3,min_periods=3).mean()
        # DP = pd.rolling_mean(fso,window=3,min_periods=3)

        # # Create DataFrame to help visualize all variables in plots
        # df_FSO = pd.concat([fso, DP, price, sma, highest_high, lowest_low], axis=1)
        # df_FSO.columns = ['fso', 'DP', 'price', 'sma', 'highest_high', 'lowest_low']
        # return df_FSO

        # For project 8, indicators must return a single results vector
        fso.columns = ['fso']
        return fso



    def run(self):

        ticker = "JPM"
        start_date = dt.datetime(2008, 1, 1)
        end_date = dt.datetime(2009, 12, 31)
        start_value = 100000
        lookback = 20


        ## FIGURE 1: %B for 'JPM'
        # Implement bbp
        # Good reference: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_perce
        df_bbp = self.PercentB(lookback)



        ## FIGURE 2: Price/EMA for 'JPM'
        # Implement Price/EMA
        df_price_EMA = self.price_EMA(lookback=50)



        ## FIGURE 3: RSI for 'JPM'
        # Implement RSI
        df_rsi = self.RSI(lookback)



        ## FIGURE 4: ROC for 'JPM'
        # Implement ROC
        df_roc = self.ROC(lookback=12)





        ## FIGURE 5: FSO for 'JPM'
        # Implement FSO
        df_FSO = self.FSO(lookback=14)




def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jmoreno62"


if __name__ == "__main__":
    ticker = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_value = 100000
    lookback = 20

    JPM = indicators(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)

    JPM.run()


