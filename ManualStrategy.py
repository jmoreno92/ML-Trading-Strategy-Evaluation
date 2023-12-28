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
import numpy as np
import matplotlib.pyplot as plt

import util as ut
import indicators as ind

# import TheoreticallyOptimalStrategy as tos
import marketsimcode as mktsim

  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
class ManualStrategy(object):
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    Code implementing a ManualStrategy object (your Manual Strategy) in the strategy_evaluation/ directory.
    It should implement testPolicy() which returns a trades data frame
  		  	   		  	  		  		  		    	 		 		   		 		  
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


    def testPolicy(
        self, symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Tests your policy implementing set of rules using indicators.  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you are trading, defaults to 'AAPL'		  	   		  	  		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2010  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 31/12/2011  		  	   		  	  		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  	  		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  	  		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  	  		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        # Note: this code is based on the VectorizeMe example shown in class.

        # Verify start and end dates are trading dates
        sd = self.TradingDay(symbol,sd, type='start')
        ed = self.TradingDay(symbol, ed, type='end')

        ### INDICATORS
        # Start indicators using symbol as reference
        indicator = ind.indicators(symbol=symbol, sd=sd, ed=ed, sv=sv)
        price_EMA = indicator.price_EMA(lookback=50)
        bbp = indicator.PercentB(lookback=20)
        # rsi = indicator.RSI(lookback=20)
        roc = indicator.ROC(lookback=12)
        # fso = indicator.FSO(lookback=14)  # vectorize or delete

        # indicators using SPY as reference
        SPY_ind = ind.indicators(symbol='SPY', sd=sd, ed=ed, sv=sv)
        # spy_bbp = SPY_ind.PercentB(lookback=20)
        spy_price_EMA = SPY_ind.price_EMA(lookback=50)
        # spy_rsi = SPY_ind.RSI(lookback=20)
        # spy_roc = SPY_ind.ROC(lookback=12)

        # Rename column name of indicators to match symbol. This is required for vectorization when setting
        # conditionals for signal dataframe
        price_EMA.columns = [symbol]
        bbp.columns = [symbol]
        # rsi.columns = [symbol]
        roc.columns = [symbol]
        # fso.columns = [symbol]
        spy_price_EMA.columns = [symbol]

        # Get prices of symbol
        price = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=False)


        ### TRADING STRATEGY
        ### Use indicators for manual trading strategy

        # Signal starts as NaN array of the same shape/index as price.
        signal = price.copy()
        signal.ix[:, :] = np.NaN

        # Create a binary (0-1) array showing when price is above EMA=-50
        ema_cross = pd.DataFrame(0, index=price_EMA.index, columns=price_EMA.columns)
        ema_cross[price_EMA >= 1] = 1

        # Turn that array into one that only shows the crossing (-1 == cross down, +1 == cross up)
        ema_cross[1:] = ema_cross.diff()
        ema_cross.ix[0] = 0

        # Apply our entry signal conditions all at once.
        # signal[(price_EMA < 0.95) & (bbp < 0) & ((rsi < 30) | (roc < -15)) & (spy_price_EMA < 0.95) ] = 1    # Long position
        # signal[(price_EMA > 1.05) & (bbp > 1) & ((rsi > 70) | (roc > 15)) & (spy_price_EMA > 1.05) ] = -1   # Short position

        signal[(price_EMA < 0.95) & (bbp < 0) & (roc < -5) & (spy_price_EMA < 0.95) ] = 1    # Long position
        signal[(price_EMA > 1.05) & (bbp > 1) & (roc > 5) & (spy_price_EMA > 1.05) ] = -1   # Short position


        # Apply our exit signal conditions all at once. Again, this represents TARGET SHARES
        signal[(ema_cross != 0)] = 0    # Out position

        # Try
        # signal.ffill(inplace=True)
        # signal.fillna(0, inplace=True)
        # signal = signal.diff()
        # signal.ix[0] = 0


        signal = signal.dropna(axis=0)  # drop np.nan rows in signal



        # Convert signal dataframe to trades. Allowable positions are 1000 long, 1000 short, or 0 shares.
        trades = signal.copy()
        position = 0
        for i in range(len(trades)):
            if trades.iloc[i,0] > 0:    # Long position
                trades.iloc[i,0] = min(2000, 1000 - position)   # Maximum trade using min function
                position = 1000
            elif trades.iloc[i,0] == 0: # Out position
                trades.iloc[i,0] = -position    # Trade to have position to be zero
                position = 0
            elif trades.iloc[i,0] < 0:  # Short position
                trades.iloc[i,0] = max(-2000, -1000 - position)     # Minimum trade using max function
                position = -1000

        trades = trades.loc[(trades != 0).any(axis=1)]  # Remove rows with all 0s in a Dataframe

        # Make sure first day and last day have a value. This is useful for the marketsim code.
        if sd not in trades.index:      # Add start date 'sd' to the trades dataframe, if it is not there
            trades.loc[sd] = 0
        if ed not in trades.index:      # Add end date 'ed' to the trades dataframe, if it is not there
            trades.loc[ed] = 0

        trades = trades.sort_index()    # sort trades by date

        if self.verbose:
            print("Type:\n", type(trades))  # it better be a DataFrame!
        if self.verbose:
            print("\nTrades:\n", trades)

        return trades


    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jmoreno62"


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

    # MANUAL STRATEGY
    ms = ManualStrategy(verbose=False, commission=9.95, impact=0.005)
    dftrades_ME = ms.testPolicy(symbol=ticker , sd=start_date, ed=end_date, sv=start_value)
    portvals_ME = mktsim.compute_portvals(orders_info=dftrades_ME, start_val=start_value,
                                          commission=ms.commission, impact=ms.impact)
    # Get metrics from TOS portfolio values
    cum_ret_ME, avg_daily_ret_ME, std_daily_ret_ME, sharpe_ratio_ME, final_val_ME = mktsim.assess_portfolio(
        sd=start_date,ed=end_date,syms=[],allocs=[],sv=1,gen_plot=False,dfvalue=portvals_ME)


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


    # # THEORETICALLY OPTIMAL STRATEGY (TOS)
    # dftrades_TOS = tos.testPolicy(symbol =ticker, sd=start_date, ed=end_date, sv = start_value)
    # # Get the portfolio values based on calculated trades using TOS
    # portvals_TOS = mktsim.compute_portvals(orders_info=dftrades_TOS, start_val=start_value, commission=9.95, impact=0.005)
    # # Get metrics from TOS portfolio values
    # cum_ret_TOS, avg_daily_ret_TOS, std_daily_ret_TOS, sharpe_ratio_TOS, final_val_TOS = mktsim.assess_portfolio(
    #     sd=start_date,ed=end_date,syms=[],allocs=[],sv=1,gen_plot=False,dfvalue=portvals_TOS)



    dfcompare =  pd.concat([portvals_ME, portvals_bmark], axis = 1)
    dfcompare = dfcompare.rename(columns={0: "Manual Strategy", 1: "Benchmark"})
    dfcompare.fillna(method="ffill", inplace=True)   # fill forward first
    dfcompare.fillna(method="bfill", inplace=True)  # fill backwards second
    dfcompare = dfcompare/dfcompare.iloc[0]     # Normalized portfolio values

    # Figure 1: Plot chart of Manual Strategy vs Benchmark portfolio
    fig1 = plt.figure("Figure 1", figsize=(7,4.8))   #figsize with default values
    ax = dfcompare["Manual Strategy"].plot(color="red")
    ax = dfcompare["Benchmark"].plot(color="purple")
    # ax = dfcompare["TOS"].plot(color="gray")
    ax.set_title("In-Sample Manual Strategy vs Benchmark Performance using 'JPM'", fontsize=12)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized portfolio value")
    ax.set_xticks(pd.date_range(start_date, end_date, periods=12))

    position = dftrades_ME.cumsum()
    for i in position.index: # 'for' loop for vertical lines
        if position.loc[i,ticker] == 1000:
            ax.axvline(i, color='blue', ls='--', linewidth='1') # Vertical blue line indicating LONG entry points
        elif position.loc[i,ticker] == -1000:
            ax.axvline(i, color='black', ls='--', linewidth='1')  # Vertical black line indicating SHORT entry points
        elif position.loc[i,ticker] == 0:
            ax.axvline(i, color='pink', ls='--', linewidth='1')  # Vertical black line indicating OUT entry points
        else:
            pass

    plt.savefig("images/Figure1")




    # Figure 2: Table showing to 6 digits the metrics of Benchmark vs Portfolio
    fig2 = plt.figure("Figure 2")   #figsize with default values
    dftable = pd.DataFrame(index=['Cumulative Return', 'Std. Deviation of Daily Returns', 'Mean of Daily Returns',
                                  'Sharpe Ratio', 'Final Value'],
                           columns=['Benchmark','Manual Strategy'])
    dftable.iloc[0,0] = f"{cum_ret_bmark:.6f}"
    dftable.iloc[1, 0] = f"{std_daily_ret_bmark:.6f}"
    dftable.iloc[2,0] = f"{avg_daily_ret_bmark:.6f}"
    dftable.iloc[3, 0] = f"{sharpe_ratio_bmark:.6f}"
    dftable.iloc[4, 0] = f"{final_val_bmark:.2f}"


    dftable.iloc[0,1] = f"{cum_ret_ME:.6f}"
    dftable.iloc[1, 1] = f"{std_daily_ret_ME:.6f}"
    dftable.iloc[2,1] = f"{avg_daily_ret_ME:.6f}"
    dftable.iloc[3, 1] = f"{sharpe_ratio_ME:.6f}"
    dftable.iloc[4, 1] = f"{final_val_ME:.2f}"


    # Solution from https://stackoverflow.com/questions/35634238/how-to-save-a-pandas-dataframe-table-as-a-png
    fig2, ax2 = plt.subplots(figsize=(10, 2))  # set size frame
    ax2.set_title("In-Sample Manual Strategy vs Benchmark Metrics using 'JPM'", fontsize=12)
    ax2.xaxis.set_visible(False)  # hide the x axis
    ax2.yaxis.set_visible(False)  # hide the y axis
    ax2.set_frame_on(False)  # no visible frame
    summary = pd.plotting.table(ax2, dftable, loc='upper right', colWidths=[0.17] * len(dftable.columns))  # where df is your data frame
    summary.auto_set_font_size(False)  # Activate set fontsize manually
    summary.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
    summary.scale(1.2, 1.2)  # change size table
    # plt.savefig("images/Figure2")
    plt.savefig("images/Table1a")


def OutSampleRun():

    ## OUT-OF-SAMPLE PERIOD
    ticker = 'JPM'
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 31)    # Last trading day
    start_value = 100000

    # MANUAL STRATEGY
    ms = ManualStrategy(verbose=False, commission=9.95, impact=0.005)
    dftrades_ME = ms.testPolicy(symbol=ticker , sd=start_date, ed=end_date, sv=start_value)
    portvals_ME = mktsim.compute_portvals(orders_info=dftrades_ME, start_val=start_value,
                                          commission=ms.commission, impact=ms.impact)
    # Get metrics from TOS portfolio values
    cum_ret_ME, avg_daily_ret_ME, std_daily_ret_ME, sharpe_ratio_ME, final_val_ME = mktsim.assess_portfolio(
        sd=start_date,ed=end_date,syms=[],allocs=[],sv=1,gen_plot=False,dfvalue=portvals_ME)


    # BENCHMARK
    # Create dataframe for benchmark. Performance of portfolio investing 1000 shares of JPM and holding that position
    # Important: Update start day and end date to consider first and last trading day in the year
    dftrades_bmark = pd.DataFrame(index=[dt.datetime(2010, 1, 4),dt.datetime(2011, 12, 30)], columns=[ticker])   # there are no trades on Jan 1
    dftrades_bmark[ticker][0] = 1000   #investing 1000 shares and holding that position
    dftrades_bmark[ticker][1] = 0       #Placeholder to keep the end_date in dftrades_bmark
    # Get the portfolio values based on benchmark trade
    portvals_bmark = mktsim.compute_portvals(orders_info=dftrades_bmark, start_val=start_value, commission=9.95, impact=0.005)
    # Get metrics from TOS portfolio values
    cum_ret_bmark, avg_daily_ret_bmark, std_daily_ret_bmark, sharpe_ratio_bmark, final_val_bmark = mktsim.assess_portfolio(
        sd=start_date,ed=end_date,syms=[],allocs=[],sv=1,gen_plot=False,dfvalue=portvals_bmark)


    # # THEORETICALLY OPTIMAL STRATEGY (TOS)
    # dftrades_TOS = tos.testPolicy(symbol =ticker, sd=start_date, ed=end_date, sv = start_value)
    # # Get the portfolio values based on calculated trades using TOS
    # portvals_TOS = mktsim.compute_portvals(orders_info=dftrades_TOS, start_val=start_value, commission=9.95, impact=0.005)
    # # Get metrics from TOS portfolio values
    # cum_ret_TOS, avg_daily_ret_TOS, std_daily_ret_TOS, sharpe_ratio_TOS, final_val_TOS = mktsim.assess_portfolio(
    #     sd=start_date,ed=end_date,syms=[],allocs=[],sv=1,gen_plot=False,dfvalue=portvals_TOS)



    dfcompare =  pd.concat([portvals_ME, portvals_bmark], axis = 1)
    dfcompare = dfcompare.rename(columns={0: "Manual Strategy", 1: "Benchmark"})
    dfcompare.fillna(method="ffill", inplace=True)   # fill forward first
    dfcompare.fillna(method="bfill", inplace=True)  # fill backwards second
    dfcompare = dfcompare/dfcompare.iloc[0]     # Normalized portfolio values

    # Figure 3: Plot chart of Manual Strategy vs Benchmark portfolio
    fig3 = plt.figure("Figure 3", figsize=(7,4.8))   #figsize with default values
    ax = dfcompare["Manual Strategy"].plot(color="red")
    ax = dfcompare["Benchmark"].plot(color="purple")
    # ax = dfcompare["TOS"].plot(color="gray")
    ax.set_title("Out-of-Sample Manual Strategy vs Benchmark Performance using 'JPM'", fontsize=12)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized portfolio value")
    ax.set_xticks(pd.date_range(start_date, end_date, periods=12))

    position = dftrades_ME.cumsum()
    for i in position.index: # 'for' loop for vertical lines
        if position.loc[i,ticker] == 1000:
            ax.axvline(i, color='blue', ls='--', linewidth='1') # Vertical blue line indicating LONG entry points
        elif position.loc[i,ticker] == -1000:
            ax.axvline(i, color='black', ls='--', linewidth='1')  # Vertical black line indicating SHORT entry points
        elif position.loc[i,ticker] == 0:
            ax.axvline(i, color='pink', ls='--', linewidth='1')  # Vertical black line indicating OUT entry points
        else:
            pass

    # plt.savefig("images/Figure3")
    plt.savefig("images/Figure2")




    # Figure 4: Table showing to 6 digits the metrics of Benchmark vs Portfolio
    fig4 = plt.figure("Figure 4")   #figsize with default values
    dftable = pd.DataFrame(index=['Cumulative Return', 'Std. Deviation of Daily Returns', 'Mean of Daily Returns',
                                  'Sharpe Ratio', 'Final Value'],
                           columns=['Benchmark','Manual Strategy'])
    dftable.iloc[0,0] = f"{cum_ret_bmark:.6f}"
    dftable.iloc[1, 0] = f"{std_daily_ret_bmark:.6f}"
    dftable.iloc[2,0] = f"{avg_daily_ret_bmark:.6f}"
    dftable.iloc[3, 0] = f"{sharpe_ratio_bmark:.6f}"
    dftable.iloc[4, 0] = f"{final_val_bmark:.2f}"


    dftable.iloc[0,1] = f"{cum_ret_ME:.6f}"
    dftable.iloc[1, 1] = f"{std_daily_ret_ME:.6f}"
    dftable.iloc[2,1] = f"{avg_daily_ret_ME:.6f}"
    dftable.iloc[3, 1] = f"{sharpe_ratio_ME:.6f}"
    dftable.iloc[4, 1] = f"{final_val_ME:.2f}"


    # Solution from https://stackoverflow.com/questions/35634238/how-to-save-a-pandas-dataframe-table-as-a-png
    fig4, ax4 = plt.subplots(figsize=(10, 2))  # set size frame
    ax4.set_title("Out-of-Sample Manual Strategy vs Benchmark Metrics using 'JPM'", fontsize=12)
    ax4.xaxis.set_visible(False)  # hide the x axis
    ax4.yaxis.set_visible(False)  # hide the y axis
    ax4.set_frame_on(False)  # no visible frame
    summary = pd.plotting.table(ax4, dftable, loc='upper right', colWidths=[0.17] * len(dftable.columns))  # where df is your data frame
    summary.auto_set_font_size(False)  # Activate set fontsize manually
    summary.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
    summary.scale(1.2, 1.2)  # change size table
    # plt.savefig("images/Figure4")
    plt.savefig("images/Table1b")


if __name__ == "__main__":

    InSampleRun()
    OutSampleRun()

