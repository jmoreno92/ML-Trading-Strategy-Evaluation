""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  	  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  	  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  	  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  	  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  		  		  		    	 		 		   		 		  
or edited.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  	  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  	  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Student Name: Jorge Salvador Aguilar Moreno  		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jmoreno62	  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845924		  	  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  


import datetime as dt
import pandas as pd
import random as rand
import numpy as np
import matplotlib.pyplot as plt

import util as ut
import indicators as ind

# import TheoreticallyOptimalStrategy as tos
import marketsimcode as mktsim
import QLearner as ql
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  	  		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  	  		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    # constructor  		  	   		  	  		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		  	  		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		  	  		  		  		    	 		 		   		 		  
        self.commission = commission  		  	   		  	  		  		  		    	 		 		   		 		  
        ### INITIALIZE Q-LEARNER
        self.Qlearner = ql.QLearner(num_states=10000,num_actions=3,alpha=0.2,gamma=0.9,
                                    rar=0.98,radr=0.999,dyna=0,verbose=False)  # initialize the learner
        self.threshold = 0  # threshold to be overridden in add_evidence function


    def add_evidence(
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        symbol="AAPL",
        sd=dt.datetime(2008, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
    ):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		  	  		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        # this method should create a QLearner, and train it for trading
  		  	   		  	  		  		  		    	 		 		   		 		  
        # add your code to do learning here

        # Verify start and end dates are trading dates
        sd = self.TradingDay(symbol,sd, type='start')
        ed = self.TradingDay(symbol, ed, type='end')

        ### INDICATORS
        # Start indicators using symbol as reference
        indicator = ind.indicators(symbol=symbol, sd=sd, ed=ed, sv=sv)
        price_EMA = indicator.price_EMA(lookback=50)
        bbp = indicator.PercentB(lookback=20)
        roc = indicator.ROC(lookback=12)
        # indicators using SPY as reference
        SPY_ind = ind.indicators(symbol='SPY', sd=sd, ed=ed, sv=sv)
        spy_price_EMA = SPY_ind.price_EMA(lookback=50)
        spy_price_EMA.columns = ['SPY_price/ema']

        df_ind = pd.concat([price_EMA, bbp, roc, spy_price_EMA], axis=1)    # Concatenate indicators in dataframe
        del price_EMA, bbp, roc, spy_price_EMA, indicator, SPY_ind  # Delete vectors, as they are now part of df_ind
        self.threshold = self.createBins(df_ind, step=9)    # Identify thresholds by creating bins. Save array in learner


        # Initialize variables
        count = 0
        countlim = 20
        start = 0
        total_return = pd.DataFrame(columns=['Return'], index=df_ind.index)
        total_return.loc[:,:] = 0
        prev_total_return = total_return.copy()
        prev_total_return.loc[:, :] = 0.1


        while (count < countlim) & (total_return.iloc[-1,0] != prev_total_return.iloc[-1,0]):
        # while (count < countlim):     # Simple while statement that does not account for convergence
            # Exit condition, when total return value does not change after iteration
            prev_total_return = total_return.copy()
            total_return.loc[:,:] = 0
            position = 0
            cash = sv
            total_cost = 0
            fee = 0


            X = df_ind.iloc[start,:]    # Indicators of the first day
            state = self.discretize(X, self.threshold)   # Obtain discretized state based on threshold array
            action = self.Qlearner.querysetstate(state)   # Set the state and get first action
            # 2: Buy, 1: Do nothing, 0: Sell
            reward = 0  # reward is zero for 1st day


            # Return for first day
            sign = action - 1  # Calculate sign of action, 1: Buy, 0: Do nothing, -1: Sell
            if sign > 0:  # Long position
                trade = min(2000, 1000 - position)  # Maximum trade using min function
                position = 1000
            elif sign == 0:  # Out position
                trade = -position  # Trade to have position to be zero
                position = 0
            elif sign < 0:  # Short position
                trade = max(-2000, -1000 - position)  # Minimum trade using max function
                position = -1000

            first_date = df_ind.index[0]
            prices = self.getPrice(symbol, first_date, first_date, lookback=1)
            total_cost = abs(trade) * prices.loc[first_date, symbol]
            fee = abs(np.sign(trade)) * (self.commission + self.impact * total_cost) # fee increments every transaction
            cash = cash - (trade * prices.loc[first_date, symbol]) - fee
            total_return.loc[first_date,:] = cash + (position * prices.loc[first_date, symbol])



            for date in df_ind.index[1:]:
                prices = self.getPrice(symbol,date,date,lookback=1)
                reward = self.calcReward(action,trade,prices)    # reward method considers impact
                action = self.Qlearner.query(state,reward)    # 2: Buy, 1: Do nothing, 0: Sell
                sign = action - 1   # Calculate sign of action, 1: Buy, 0: Do nothing, -1: Sell

                # Total return  =  starting cash  +  position - fees
                if sign > 0:  # Long position
                    trade = min(2000, 1000 - position)   # Maximum trade using min function
                    position = 1000
                elif sign == 0: # Out position
                    trade = -position    # Trade to have position to be zero
                    position = 0
                elif sign < 0:      # Short position
                    trade = max(-2000, -1000 - position)     # Minimum trade using max function
                    position = -1000

                total_cost = abs(trade) * prices.loc[date,symbol]
                fee = abs(np.sign(trade)) * (self.commission + self.impact * total_cost) # fee increments every transaction
                cash = cash - (trade * prices.loc[date,symbol]) - fee
                total_return.loc[date,:]  = cash + (position * prices.loc[date,symbol])


                X = df_ind.loc[date, :]
                state = self.discretize(X, self.threshold)  # Obtain discretized state based on threshold array

            count += 1

            if self.verbose:
                print(count)
                print(total_return.iloc[-1,0] )

        if self.verbose:
            print("ADD EVIDENCE")
            print(total_return)

            if count == countlim:
                print("Time limit")
            else:
                print("Learner convergence")

            # PSEUDO CODE FOR ADD EVIDENCE
            # X = indicators
            # querysetstate(X)
            # For each day:
            #   r = calculate
            #   a = query(X,r)
            #   Implement action: e.g. go long, simulate buying 1000 shares and when step forward in next day, calculate reward depending on what I am holding
            #   add action to trades dataframe
            #   X = new state
            # Check if converged.


    def testPolicy(
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        symbol="AAPL",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100000,
    ):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		  	  		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  	  		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  	  		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  	  		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        # this method should use the existing policy and test it against new data

        # Verify start and end dates are trading dates
        sd = self.TradingDay(symbol,sd, type='start')
        ed = self.TradingDay(symbol, ed, type='end')

        ### TRADES DATAFRAME BASED ON PRICES
        trades = self.getPrice(symbol, sd, ed, lookback = 0)
        trades.ix[:, :] = np.NaN


        ### INDICATORS
        # Start indicators using symbol as reference
        indicator = ind.indicators(symbol=symbol, sd=sd, ed=ed, sv=sv)
        price_EMA = indicator.price_EMA(lookback=50)
        bbp = indicator.PercentB(lookback=20)
        roc = indicator.ROC(lookback=12)
        # indicators using SPY as reference
        SPY_ind = ind.indicators(symbol='SPY', sd=sd, ed=ed, sv=sv)
        spy_price_EMA = SPY_ind.price_EMA(lookback=50)
        spy_price_EMA.columns = ['SPY_price/ema']

        df_ind = pd.concat([price_EMA, bbp, roc, spy_price_EMA], axis=1)    # Concatenate indicators in dataframe
        del price_EMA, bbp, roc, spy_price_EMA, indicator, SPY_ind  # Delete vectors, as they are now part of df_ind
        # threshold from in-sample data is stored in self.


        start = 0
        position = 0

        if self.verbose:    # Part 1 of 4 to get returns
            count = 0
            countlim = 25
            start = 0
            total_return = pd.DataFrame(columns=['Return'], index=df_ind.index)
            total_return.loc[:, :] = 0
            prev_total_return = total_return.copy()
            prev_total_return.loc[:, :] = 0.1
            prev_total_return = total_return.copy()
            total_return.loc[:, :] = 0
            position = 0
            cash = sv
            total_cost = 0
            fee = 0



        X = df_ind.iloc[start,:]    # Indicators of the first day
        state = self.discretize(X, self.threshold)   # Obtain discretized state based on threshold array
        action = self.Qlearner.querysetstate(state)   # Set the state and get first action
        # 2: Buy, 1: Do nothing, 0: Sell
        sign = action - 1  # Calculate sign of action, 1: Buy, 0: Do nothing, -1: Sell
        # Implement action
        if sign > 0:  # Long position
            trades.iloc[0, 0] = min(2000, 1000 - position)  # Maximum trade using min function
            position = 1000
        elif sign == 0:  # Out position
            trades.iloc[0, 0]= -position  # Trade to have position to be zero
            position = 0
        elif sign < 0:  # Short position
            trades.iloc[0, 0]= max(-2000, -1000 - position)  # Minimum trade using max function
            position = -1000

        if self.verbose:    # Part 2 of 4 to get returns
            first_date = df_ind.index[0]
            prices = self.getPrice(symbol, first_date, first_date, lookback=1)
            total_cost = abs(trades.iloc[0, 0]) * prices.loc[first_date, symbol]
            fee = abs(np.sign(trades.iloc[0, 0])) * (self.commission + self.impact * total_cost)  # fee increments every transaction
            cash = cash - (trades.iloc[0, 0] * prices.loc[first_date, symbol]) - fee
            total_return.loc[first_date, :] = cash + (position * prices.loc[first_date, symbol])


        for date in df_ind.index[1:]:
            prices = self.getPrice(symbol,date,date,lookback=1)
            action = self.Qlearner.querysetstate(state) # # 2: Buy, 1: Do nothing, 0: Sell
            sign = action - 1   # Calculate sign of action, 1: Buy, 0: Do nothing, -1: Sell
            # Implement action
            if sign > 0:    # Long position
                trades.loc[date,symbol] = min(2000, 1000 - position)   # Maximum trade using min function
                position = 1000
            elif sign == 0: # Out position
                trades.loc[date,symbol] = -position    # Trade to have position to be zero
                position = 0
            elif sign < 0:  # Short position
                trades.loc[date,symbol] = max(-2000, -1000 - position)     # Minimum trade using max function
                position = -1000

            if self.verbose:    # Part 3 of 4 to get returns
                total_cost = abs(trades.loc[date,symbol])  * prices.loc[date, symbol]
                fee = abs(np.sign(trades.loc[date,symbol] )) * (
                            self.commission + self.impact * total_cost)  # fee increments every transaction
                cash = cash - (trades.loc[date,symbol]  * prices.loc[date, symbol]) - fee
                total_return.loc[date, :] = cash + (position * prices.loc[date, symbol])


            X = df_ind.loc[date, :]
            state = self.discretize(X, self.threshold)  # Obtain discretized state based on threshold array

        trades = trades.loc[(trades != 0).any(axis=1)]  # Remove rows with all 0s in a Dataframe


        # Make sure first day and last day have a value. This is useful for the marketsim code.
        if sd not in trades.index:      # Add start date 'sd' to the trades dataframe, if it is not there
            trades.loc[sd] = 0
        if ed not in trades.index:      # Add end date 'ed' to the trades dataframe, if it is not there
            trades.loc[ed] = 0


        if self.verbose:    # Part 4 of 4 to get returns
            print("TEST POLICY")
            print(total_return)

            print(type(trades))  # it better be a DataFrame!
            print(trades)
        return trades


    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jmoreno62"


    def createBins(self,df,step=9):
        """
        Function takes dataframe and for each column, it creates an array with equally spaced bins based on steps.

        :param df: dataframe with indicators as columns
        :type df: pandas dataframe
        :param step: step of number of bins to be created
        :type step: int
        :return: threshold of limits for each indicator
        :rtype: numpy array, columns = number of indicators, rows = steps
        """
        NoRows, NoCols = df.shape
        stepsize = df.shape[0]/ step    # Step size based on number of rows
        df = df.reset_index(drop=True)  # drop index
        # Sort columns individually and have it sorted
        # Reference: https://stackoverflow.com/questions/24171511/pandas-sort-each-column-individually
        df_sorted = pd.concat([df[col].sort_values().reset_index(drop=True) for col in df], axis=1, ignore_index=True)

        threshold = np.zeros((step,NoCols))
        for col in df_sorted:
            for i in range(step):
                threshold[i,col] = df_sorted.loc[int( (i+1)*stepsize - 1 ) , col]

        return threshold


    def discretize(self,s_ind, threshold):
        """
        Creates integer to represent 'state' based on Series of indicators, and threshold araray

        :param s_ind: indicators for a single day
        :type s_ind: Series
        :param threshold: matrix of threshold limits
        :type threshold: numpy array
        :return: state as integer
        :rtype: int
        """
        Factor = 10 ** (threshold.shape[1] -1)
        s = 0
        for i in range(len(s_ind)):
            # Reference: # https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
            number = np.digitize(s_ind[i], threshold[:,i])
            s = s + Factor * number
            Factor /= 10
        return int(s)


    def getPrice(self, symbol, sd, ed, lookback = 1):
        add_day = 0
        prev_days = 0
        price = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=True)

        if lookback == 0:   # If we do not have a lookback period
            if 'SPY' not in symbol:  # Once we have the trading days, drop SPY if it is not in portfolio
                price = price.drop(columns=['SPY'])
            price.fillna(method="ffill", inplace=True)  # fill forward first
            price.fillna(method="bfill", inplace=True)  # fill backwards second
            return price

        else:   # If we have a lookback period
            # Extended price 'ext_price' DataFrame to account for trading days before start day equal to lookback
            while prev_days < lookback:
                ext_price = ut.get_data([symbol], pd.date_range(sd - dt.timedelta(days=lookback+add_day), ed), addSPY=True)
                add_day += 1
                prev_days = len(ext_price) - len(price)
            del add_day,prev_days

            if 'SPY' not in symbol:  # Once we have the trading days, drop SPY if it is not in portfolio
                price = price.drop(columns=['SPY'])
                ext_price = ext_price.drop(columns=['SPY'])
            ext_price.fillna(method="ffill", inplace=True)  # fill forward first
            ext_price.fillna(method="bfill", inplace=True)  # fill backwards second
            return ext_price


    def calcReward(self,action,trade,prices):
        # Calculate reward as percent change
        sign = action - 1     # Calculate sign reward, 1: Buy, 0: Do nothing, -1: Sell
        total_cost = abs(trade) * prices.iloc[1,0]
        fee = abs(np.sign(trade)) * (self.commission + self.impact * total_cost)
        # reward = sign * (prices.iloc[1,0]/prices.iloc[0,0] - 1)   # Original, where reward is not penalized by impact
        reward = sign * ( ( prices.iloc[1,0] * 1000 - sign * fee) / ( prices.iloc[0,0] * 1000 ) - 1 )
        return reward


    def TradingDay(self,symbol,date, type='start'):
        add_date = 0
        price = ut.get_data([symbol], pd.date_range(date, date), addSPY=True)

        while (type == 'start') & (date not in price.index):
            add_date += 1
            date = date + dt.timedelta(days= add_date)
            price = ut.get_data([symbol], pd.date_range(date, date),addSPY=True)

        while (type == 'end') & (date not in price.index):
            add_date += 1
            date = date - dt.timedelta(days= add_date)
            price = ut.get_data([symbol], pd.date_range(date, date),addSPY=True)

        return date



def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jmoreno62"


def InSampleRun():

    ## IN-SAMPLE PERIOD
    ticker = 'JPM'
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_value = 100000

    # rand.seed(5)

    # STRATEGY LEARNER

    learner = StrategyLearner(verbose=False, commission=9.95, impact=0.005)     # constructor
    # learner = StrategyLearner(verbose=False, commission=0, impact=0.000)  # constructor without impact
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)     # training phase
    dftrades_SL = learner.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)      # testing phase
    portvals_SL = mktsim.compute_portvals(orders_info=dftrades_SL, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL, avg_daily_ret_SL, std_daily_ret_SL, sharpe_ratio_SL, final_val_SL = mktsim.assess_portfolio(
        sd=start_date, ed=end_date, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL)


    # BENCHMARK
    # Create dataframe for benchmark. Performance of portfolio investing 1000 shares of JPM and holding that position
    # Important: Update start day to consider first trading day in the year
    dftrades_bmark = pd.DataFrame(index=[dt.datetime(2008, 1, 2),end_date], columns=[ticker])   # there are no trades on Jan 1
    dftrades_bmark[ticker][0] = 1000   #investing 1000 shares and holding that position
    dftrades_bmark[ticker][1] = 0       #Placeholder to keep the end_date in dftrades_bmark
    # Get the portfolio values based on benchmark trade
    portvals_bmark = mktsim.compute_portvals(orders_info=dftrades_bmark, start_val=start_value, commission=9.95, impact=0.005)
    # Get metrics from TOS portfolio values
    cum_ret_bmark, avg_daily_ret_bmark, std_daily_ret_bmark, sharpe_ratio_bmark, final_val_bmark = mktsim.assess_portfolio(
        sd=start_date,ed=end_date,syms=[],allocs=[],sv=1,gen_plot=False,dfvalue=portvals_bmark)



    dfcompare =  pd.concat([portvals_SL, portvals_bmark], axis = 1)
    dfcompare = dfcompare.rename(columns={0: "Q Learner", 1: "Benchmark"})
    dfcompare.fillna(method="ffill", inplace=True)   # fill forward first
    dfcompare.fillna(method="bfill", inplace=True)  # fill backwards second
    dfcompare = dfcompare/dfcompare.iloc[0]     # Normalized portfolio values

    # Figure 5: Plot chart of Manual Strategy vs Benchmark portfolio
    fig5 = plt.figure("Figure 5", figsize=(7,4.8))   #figsize with default values
    ax = dfcompare["Q Learner"].plot(color="red")
    ax = dfcompare["Benchmark"].plot(color="purple")
    # ax = dfcompare["TOS"].plot(color="gray")
    ax.set_title("In-Sample Q Learner vs Benchmark Performance using 'JPM'", fontsize=12)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized portfolio value")
    ax.set_xticks(pd.date_range(start_date, end_date, periods=12))

    # position = dftrades_SL.cumsum()
    # for i in position.index: # 'for' loop for vertical lines
    #     if position.loc[i,ticker] == 1000:
    #         ax.axvline(i, color='blue', ls='--', linewidth='1') # Vertical blue line indicating LONG entry points
    #     elif position.loc[i,ticker] == -1000:
    #         ax.axvline(i, color='black', ls='--', linewidth='1')  # Vertical black line indicating SHORT entry points
    #     elif position.loc[i,ticker] == 0:
    #         ax.axvline(i, color='pink', ls='--', linewidth='1')  # Vertical black line indicating OUT entry points
    #     else:
    #         pass

    plt.savefig("images/Figure5")




    # Figure 6: Table showing to 6 digits the metrics of Benchmark vs Portfolio
    fig6 = plt.figure("Figure 6")   #figsize with default values
    dftable = pd.DataFrame(index=['Cumulative Return', 'Std. Deviation of Daily Returns', 'Mean of Daily Returns',
                                  'Sharpe Ratio', 'Final Value'],
                           columns=['Benchmark','Q Learner'])
    dftable.iloc[0,0] = f"{cum_ret_bmark:.6f}"
    dftable.iloc[1, 0] = f"{std_daily_ret_bmark:.6f}"
    dftable.iloc[2,0] = f"{avg_daily_ret_bmark:.6f}"
    dftable.iloc[3, 0] = f"{sharpe_ratio_bmark:.6f}"
    dftable.iloc[4, 0] = f"{final_val_bmark:.2f}"


    dftable.iloc[0,1] = f"{cum_ret_SL:.6f}"
    dftable.iloc[1, 1] = f"{std_daily_ret_SL:.6f}"
    dftable.iloc[2,1] = f"{avg_daily_ret_SL:.6f}"
    dftable.iloc[3, 1] = f"{sharpe_ratio_SL:.6f}"
    dftable.iloc[4, 1] = f"{final_val_SL:.2f}"


    # Solution from https://stackoverflow.com/questions/35634238/how-to-save-a-pandas-dataframe-table-as-a-png
    fig6, ax6 = plt.subplots(figsize=(10, 2))  # set size frame
    ax6.set_title("In-Sample Q-Learner vs Benchmark Metrics using 'JPM'", fontsize=12)
    ax6.xaxis.set_visible(False)  # hide the x axis
    ax6.yaxis.set_visible(False)  # hide the y axis
    ax6.set_frame_on(False)  # no visible frame
    summary = pd.plotting.table(ax6, dftable, loc='upper right', colWidths=[0.17] * len(dftable.columns))  # where df is your data frame
    summary.auto_set_font_size(False)  # Activate set fontsize manually
    summary.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
    summary.scale(1.2, 1.2)  # change size table
    plt.savefig("images/Figure6")


def OutSampleRun():

    ticker = 'JPM'
    start_value = 100000
    ## IN-SAMPLE PERIOD
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    ## OUT-OF-SAMPLE PERIOD
    start_date_out_sample = dt.datetime(2010, 1, 1)
    end_date_out_sample = dt.datetime(2011, 12, 31)    # Last trading day


    # STRATEGY LEARNER
    learner = StrategyLearner(verbose=False, commission=9.95, impact=0.005)     # constructor
    # learner = StrategyLearner(verbose=False, commission=0, impact=0.000)  # constructor with zero impact
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)  # training phase
    dftrades_SL = learner.testPolicy(symbol=ticker, sd=start_date_out_sample, ed=end_date_out_sample, sv=start_value)  # testing phase
    portvals_SL = mktsim.compute_portvals(orders_info=dftrades_SL, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL, avg_daily_ret_SL, std_daily_ret_SL, sharpe_ratio_SL, final_val_SL = mktsim.assess_portfolio(
        sd=start_date, ed=end_date, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL)

    # BENCHMARK
    # Create dataframe for benchmark. Performance of portfolio investing 1000 shares of JPM and holding that position
    # Important: Update start day to consider first trading day in the year
    dftrades_bmark = pd.DataFrame(index=[dt.datetime(2010, 1, 4), dt.datetime(2011, 12, 30)],
                                  columns=[ticker])  # there are no trades on Jan 1
    dftrades_bmark[ticker][0] = 1000  # investing 1000 shares and holding that position
    dftrades_bmark[ticker][1] = 0  # Placeholder to keep the end_date in dftrades_bmark
    # Get the portfolio values based on benchmark trade
    portvals_bmark = mktsim.compute_portvals(orders_info=dftrades_bmark, start_val=start_value, commission=9.95,
                                             impact=0.005)
    # Get metrics from TOS portfolio values
    cum_ret_bmark, avg_daily_ret_bmark, std_daily_ret_bmark, sharpe_ratio_bmark, final_val_bmark = mktsim.assess_portfolio(
        sd=start_date, ed=end_date, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_bmark)

    dfcompare = pd.concat([portvals_SL, portvals_bmark], axis=1)
    dfcompare = dfcompare.rename(columns={0: "Q Learner", 1: "Benchmark"})
    dfcompare.fillna(method="ffill", inplace=True)  # fill forward first
    dfcompare.fillna(method="bfill", inplace=True)  # fill backwards second
    dfcompare = dfcompare / dfcompare.iloc[0]  # Normalized portfolio values


    # Figure 7: Plot chart of Manual Strategy vs Benchmark portfolio
    fig7 = plt.figure("Figure 7", figsize=(7, 4.8))  # figsize with default values
    ax = dfcompare["Q Learner"].plot(color="red")
    ax = dfcompare["Benchmark"].plot(color="purple")
    # ax = dfcompare["TOS"].plot(color="gray")
    ax.set_title("Out-of-Sample Q Learner vs Benchmark Performance using 'JPM'", fontsize=12)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized portfolio value")
    ax.set_xticks(pd.date_range(start_date_out_sample, end_date_out_sample, periods=12))

    # position = dftrades_SL.cumsum()
    # for i in position.index:  # 'for' loop for vertical lines
    #     if position.loc[i, ticker] == 1000:
    #         ax.axvline(i, color='blue', ls='--', linewidth='1')  # Vertical blue line indicating LONG entry points
    #     elif position.loc[i, ticker] == -1000:
    #         ax.axvline(i, color='black', ls='--', linewidth='1')  # Vertical black line indicating SHORT entry points
    #     elif position.loc[i, ticker] == 0:
    #         ax.axvline(i, color='pink', ls='--', linewidth='1')  # Vertical black line indicating OUT entry points
    #     else:
    #         pass

    plt.savefig("images/Figure7")

    # Figure 8: Table showing to 6 digits the metrics of Benchmark vs Portfolio
    fig8 = plt.figure("Figure 8")  # figsize with default values
    dftable = pd.DataFrame(index=['Cumulative Return', 'Std. Deviation of Daily Returns', 'Mean of Daily Returns',
                                  'Sharpe Ratio', 'Final Value'],
                           columns=['Benchmark', 'Q Learner'])
    dftable.iloc[0, 0] = f"{cum_ret_bmark:.6f}"
    dftable.iloc[1, 0] = f"{std_daily_ret_bmark:.6f}"
    dftable.iloc[2, 0] = f"{avg_daily_ret_bmark:.6f}"
    dftable.iloc[3, 0] = f"{sharpe_ratio_bmark:.6f}"
    dftable.iloc[4, 0] = f"{final_val_bmark:.2f}"

    dftable.iloc[0, 1] = f"{cum_ret_SL:.6f}"
    dftable.iloc[1, 1] = f"{std_daily_ret_SL:.6f}"
    dftable.iloc[2, 1] = f"{avg_daily_ret_SL:.6f}"
    dftable.iloc[3, 1] = f"{sharpe_ratio_SL:.6f}"
    dftable.iloc[4, 1] = f"{final_val_SL:.2f}"

    # Solution from https://stackoverflow.com/questions/35634238/how-to-save-a-pandas-dataframe-table-as-a-png
    fig8, ax8 = plt.subplots(figsize=(10, 2))  # set size frame
    ax8.set_title("Out-of-Sample Q-Learner vs Benchmark Metrics using 'JPM'", fontsize=12)
    ax8.xaxis.set_visible(False)  # hide the x axis
    ax8.yaxis.set_visible(False)  # hide the y axis
    ax8.set_frame_on(False)  # no visible frame
    summary = pd.plotting.table(ax8, dftable, loc='upper right',
                                colWidths=[0.17] * len(dftable.columns))  # where df is your data frame
    summary.auto_set_font_size(False)  # Activate set fontsize manually
    summary.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
    summary.scale(1.2, 1.2)  # change size table
    plt.savefig("images/Figure8")


if __name__ == "__main__":

    InSampleRun()
    OutSampleRun()