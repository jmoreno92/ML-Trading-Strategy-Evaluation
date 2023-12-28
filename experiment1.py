""""""
""" experiment1.py	  	   		  	  		  		  		    	 		 		   		 		  

Experiment 1 should compare the results of your manual strategy and strategy learner. It should:

Compare your Manual Strategy with your Strategy Learner in-sample trading JPM. Create a chart that shows:  

    Value of the ManualStrategy portfolio (normalized to 1.0 at the start)  
    Value of the StrategyLearner portfolio (normalized to 1.0 at the start)  
    Value of the Benchmark portfolio (normalized to 1.0 at the start)  

Compare your Manual Strategy with your Strategy Learner out-of-sample trading JPM. Create a chart that shows:  

    Value of the ManualStrategy portfolio (normalized to 1.0 at the start)  
    Value of the StrategyLearner portfolio (normalized to 1.0 at the start)  
    Value of the Benchmark portfolio (normalized to 1.0 at the start) 

The code that implements this experiment and generates the relevant charts and data should be submitted 
as experiment1.py. 


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
from indicators import indicators
import ManualStrategy as me
import StrategyLearner as sl


def InSampleRun():

    ## IN-SAMPLE PERIOD
    ticker = 'JPM'
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_value = 100000

    # rand.seed(5)

    # STRATEGY LEARNER
    learner = sl.StrategyLearner(verbose=False, commission=9.95, impact=0.005)     # constructor
    # learner = StrategyLearner(verbose=False, commission=0, impact=0.000)  # constructor without impact
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)     # training phase
    dftrades_SL = learner.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)      # testing phase
    portvals_SL = mktsim.compute_portvals(orders_info=dftrades_SL, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL, avg_daily_ret_SL, std_daily_ret_SL, sharpe_ratio_SL, final_val_SL = mktsim.assess_portfolio(
        sd=start_date, ed=end_date, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL)


    # MANUAL STRATEGY
    ms = me.ManualStrategy(verbose=False, commission=9.95, impact=0.005)
    dftrades_ME = ms.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)
    portvals_ME = mktsim.compute_portvals(orders_info=dftrades_ME, start_val=start_value,
                                          commission=ms.commission, impact=ms.impact)
    # # Get metrics from TOS portfolio values
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



    dfcompare =  pd.concat([portvals_SL, portvals_ME, portvals_bmark], axis = 1)
    dfcompare = dfcompare.rename(columns={0: "Q Learner", 1: "Manual Strategy", 2: "Benchmark"})
    dfcompare.fillna(method="ffill", inplace=True)   # fill forward first
    dfcompare.fillna(method="bfill", inplace=True)  # fill backwards second
    dfcompare = dfcompare/dfcompare.iloc[0]     # Normalized portfolio values

    # Figure 5: Plot chart of In-sample Q Learner vs Manual Strategy and Benchmark portfolio
    fig5 = plt.figure("Figure 5", figsize=(7,4.8))   #figsize with default values
    ax = dfcompare["Q Learner"].plot(color="green")
    ax = dfcompare["Manual Strategy"].plot(color="red")
    ax = dfcompare["Benchmark"].plot(color="purple")
    # ax = dfcompare["TOS"].plot(color="gray")
    ax.set_title("In-Sample Q Learner, Manual Strategy and Benchmark Performance using 'JPM'", fontsize=12)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized portfolio value")
    ax.set_xticks(pd.date_range(start_date, end_date, periods=12))

    # plt.savefig("images/Figure5")
    plt.savefig("images/Figure3")




    # Figure 6: Table showing to 6 digits the metrics of Benchmark, Strategy Learner, and Manual Strategy
    fig6 = plt.figure("Figure 6")   #figsize with default values
    dftable = pd.DataFrame(index=['Cumulative Return', 'Std. Deviation of Daily Returns', 'Mean of Daily Returns',
                                  'Sharpe Ratio', 'Final Value'],
                           columns=['Benchmark','Q Learner','Manual Strategy'])
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

    dftable.iloc[0,2] = f"{cum_ret_ME:.6f}"
    dftable.iloc[1, 2] = f"{std_daily_ret_ME:.6f}"
    dftable.iloc[2,2] = f"{avg_daily_ret_ME:.6f}"
    dftable.iloc[3, 2] = f"{sharpe_ratio_ME:.6f}"
    dftable.iloc[4, 2] = f"{final_val_ME:.2f}"



    # Solution from https://stackoverflow.com/questions/35634238/how-to-save-a-pandas-dataframe-table-as-a-png
    fig6, ax6 = plt.subplots(figsize=(10, 2))  # set size frame
    ax6.set_title("In-Sample Q-Learner, Manual Strategy and Benchmark Metrics using 'JPM'", fontsize=12)
    ax6.xaxis.set_visible(False)  # hide the x axis
    ax6.yaxis.set_visible(False)  # hide the y axis
    ax6.set_frame_on(False)  # no visible frame
    summary = pd.plotting.table(ax6, dftable, loc='upper right', colWidths=[0.17] * len(dftable.columns))  # where df is your data frame
    summary.auto_set_font_size(False)  # Activate set fontsize manually
    summary.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
    summary.scale(1.2, 1.2)  # change size table
    # plt.savefig("images/Figure6")
    plt.savefig("images/Table2a")

def OutSampleRun():

    ticker = 'JPM'
    start_value = 100000
    ## IN-SAMPLE PERIOD
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    ## OUT-OF-SAMPLE PERIOD
    start_date_out_sample = dt.datetime(2010, 1, 1)
    end_date_out_sample = dt.datetime(2011, 12, 31)    # Last trading day

    # rand.seed(5)

    # STRATEGY LEARNER
    learner = sl.StrategyLearner(verbose=False, commission=9.95, impact=0.005)     # constructor
    # learner = StrategyLearner(verbose=False, commission=0, impact=0.000)  # constructor with zero impact
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)  # training phase
    dftrades_SL = learner.testPolicy(symbol=ticker, sd=start_date_out_sample, ed=end_date_out_sample, sv=start_value)  # testing phase
    portvals_SL = mktsim.compute_portvals(orders_info=dftrades_SL, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL, avg_daily_ret_SL, std_daily_ret_SL, sharpe_ratio_SL, final_val_SL = mktsim.assess_portfolio(
        sd=start_date_out_sample, ed=end_date_out_sample, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL)


    # MANUAL STRATEGY
    ms = me.ManualStrategy(verbose=False, commission=9.95, impact=0.005)
    dftrades_ME = ms.testPolicy(symbol=ticker, sd=start_date_out_sample, ed=end_date_out_sample, sv=start_value)
    portvals_ME = mktsim.compute_portvals(orders_info=dftrades_ME, start_val=start_value,
                                          commission=ms.commission, impact=ms.impact)
    # # Get metrics from TOS portfolio values
    cum_ret_ME, avg_daily_ret_ME, std_daily_ret_ME, sharpe_ratio_ME, final_val_ME = mktsim.assess_portfolio(
        sd=start_date_out_sample,ed=end_date_out_sample,syms=[],allocs=[],sv=1,gen_plot=False,dfvalue=portvals_ME)


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
        sd=start_date_out_sample, ed=end_date_out_sample, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_bmark)



    dfcompare =  pd.concat([portvals_SL, portvals_ME, portvals_bmark], axis = 1)
    dfcompare = dfcompare.rename(columns={0: "Q Learner", 1: "Manual Strategy", 2: "Benchmark"})
    dfcompare.fillna(method="ffill", inplace=True)  # fill forward first
    dfcompare.fillna(method="bfill", inplace=True)  # fill backwards second
    dfcompare = dfcompare / dfcompare.iloc[0]  # Normalized portfolio values


    # Figure 7: Plot chart of Out-of-sample Q Learner vs Manual Strategy and Benchmark portfolio
    fig7 = plt.figure("Figure 7", figsize=(7, 4.8))  # figsize with default values
    ax = dfcompare["Q Learner"].plot(color="green")
    ax = dfcompare["Manual Strategy"].plot(color="red")
    ax = dfcompare["Benchmark"].plot(color="purple")
    # ax = dfcompare["TOS"].plot(color="gray")
    ax.set_title("Out-of-Sample Q Learner, Manual Strategy and Benchmark Performance using 'JPM'", fontsize=12)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized portfolio value")
    ax.set_xticks(pd.date_range(start_date_out_sample, end_date_out_sample, periods=12))

    # plt.savefig("images/Figure7")
    plt.savefig("images/Figure4")




    # Figure 8: Table showing to 6 digits the metrics of Benchmark, Strategy Learner, and Manual Strategy
    fig8 = plt.figure("Figure 8")  # figsize with default values
    dftable = pd.DataFrame(index=['Cumulative Return', 'Std. Deviation of Daily Returns', 'Mean of Daily Returns',
                                  'Sharpe Ratio', 'Final Value'],
                           columns=['Benchmark','Q Learner','Manual Strategy'])
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

    dftable.iloc[0,2] = f"{cum_ret_ME:.6f}"
    dftable.iloc[1, 2] = f"{std_daily_ret_ME:.6f}"
    dftable.iloc[2,2] = f"{avg_daily_ret_ME:.6f}"
    dftable.iloc[3, 2] = f"{sharpe_ratio_ME:.6f}"
    dftable.iloc[4, 2] = f"{final_val_ME:.2f}"


    # Solution from https://stackoverflow.com/questions/35634238/how-to-save-a-pandas-dataframe-table-as-a-png
    fig8, ax8 = plt.subplots(figsize=(10, 2))  # set size frame
    ax8.set_title("Out-of-Sample Q-Learner, Manual Strategy and Benchmark Metrics using 'JPM'", fontsize=12)
    ax8.xaxis.set_visible(False)  # hide the x axis
    ax8.yaxis.set_visible(False)  # hide the y axis
    ax8.set_frame_on(False)  # no visible frame
    summary = pd.plotting.table(ax8, dftable, loc='upper right',
                                colWidths=[0.17] * len(dftable.columns))  # where df is your data frame
    summary.auto_set_font_size(False)  # Activate set fontsize manually
    summary.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
    summary.scale(1.2, 1.2)  # change size table
    # plt.savefig("images/Figure8")
    plt.savefig("images/Table2b")


def test_code():
    # print("Jorge is the best")
    InSampleRun()
    OutSampleRun()


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jmoreno62"


if __name__ == "__main__":
    test_code()
