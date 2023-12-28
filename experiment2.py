""""""
""" experiment2.py	  	   		  	  		  		  		    	 		 		   		 		  

Conduct an experiment with your StrategyLearner that shows how changing the value of impact 
should affect in-sample trading behavior.

Select two metrics, and generate tests that will provide you with at least 3 measurements when trading JPM 
on the in-sample period with a commission of $0.00. Generate charts that support your tests and show your results. 

The code that implements this experiment and generates the relevant charts and data should be submitted 
as experiment2.py. 


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


    # STRATEGY LEARNERS

    # Commission = 0, Impact = 0.015
    learner = sl.StrategyLearner(verbose=False, commission=0.00, impact=0.015)     # constructor
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)     # training phase
    dftrades_SL_015 = learner.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)      # testing phase
    portvals_SL_015 = mktsim.compute_portvals(orders_info=dftrades_SL_015, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL_015, avg_daily_ret_SL_015, std_daily_ret_SL_015, sharpe_ratio_SL_015, final_val_SL_015 = mktsim.assess_portfolio(
        sd=start_date, ed=end_date, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL_015)


    # Commission = 0, Impact = 0.010
    learner = sl.StrategyLearner(verbose=False, commission=0.00, impact=0.010)     # constructor
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)     # training phase
    dftrades_SL_010 = learner.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)      # testing phase
    portvals_SL_010 = mktsim.compute_portvals(orders_info=dftrades_SL_010, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL_010, avg_daily_ret_SL_010, std_daily_ret_SL_010, sharpe_ratio_SL_010, final_val_SL_010 = mktsim.assess_portfolio(
        sd=start_date, ed=end_date, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL_010)


    # Commission = 0, Impact = 0.005
    learner = sl.StrategyLearner(verbose=False, commission=0.00, impact=0.005)     # constructor
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)     # training phase
    dftrades_SL_005 = learner.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)      # testing phase
    portvals_SL_005 = mktsim.compute_portvals(orders_info=dftrades_SL_005, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL_005, avg_daily_ret_SL_005, std_daily_ret_SL_005, sharpe_ratio_SL_005, final_val_SL_005 = mktsim.assess_portfolio(
        sd=start_date, ed=end_date, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL_005)


    # Commission = 0, Impact = 0.0025
    learner = sl.StrategyLearner(verbose=False, commission=0.00, impact=0.0025)     # constructor
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)     # training phase
    dftrades_SL_0025 = learner.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)      # testing phase
    portvals_SL_0025 = mktsim.compute_portvals(orders_info=dftrades_SL_0025, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL_0025, avg_daily_ret_SL_0025, std_daily_ret_SL_0025, sharpe_ratio_SL_0025, final_val_SL_0025 = mktsim.assess_portfolio(
        sd=start_date, ed=end_date, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL_0025)


    # Commission = 0, Impact = 0.000
    learner = sl.StrategyLearner(verbose=False, commission=0.00, impact=0.000)     # constructor
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)     # training phase
    dftrades_SL_000 = learner.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)      # testing phase
    portvals_SL_000 = mktsim.compute_portvals(orders_info=dftrades_SL_000, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL_000, avg_daily_ret_SL_000, std_daily_ret_SL_000, sharpe_ratio_SL_000, final_val_SL_000 = mktsim.assess_portfolio(
        sd=start_date, ed=end_date, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL_000)




    dfcompare =  pd.concat([portvals_SL_015, portvals_SL_010, portvals_SL_005, portvals_SL_0025, portvals_SL_000], axis = 1)
    dfcompare = dfcompare.rename(columns={0: "Impact=0.015", 1: "Impact=0.010", 2: "Impact=0.005", 3: "Impact=0.0025",
                                          4: "Impact=0.000"})
    dfcompare.fillna(method="ffill", inplace=True)   # fill forward first
    dfcompare.fillna(method="bfill", inplace=True)  # fill backwards second
    dfcompare = dfcompare/dfcompare.iloc[0]     # Normalized portfolio values

    # Figure 9: Plot chart of In-sample Q Learner portfolio considering different impact
    fig9 = plt.figure("Figure 9", figsize=(7,4.8))   #figsize with default values
    ax = dfcompare["Impact=0.015"].plot(color="teal")
    ax = dfcompare["Impact=0.010"].plot(color="blueviolet")
    ax = dfcompare["Impact=0.005"].plot(color="darkorange")
    ax = dfcompare["Impact=0.0025"].plot(color="limegreen")
    ax = dfcompare["Impact=0.000"].plot(color="red")
    # ax = dfcompare["TOS"].plot(color="gray")
    ax.set_title("Effect of Impact for In-Sample Period in Q-Learner using 'JPM'", fontsize=12)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized portfolio value")
    ax.set_xticks(pd.date_range(start_date, end_date, periods=12))

    # plt.savefig("images/Figure9")
    plt.savefig("images/Figure5")



    # Figure 10: Table showing to 6 digits the metrics of Strategy Learner with different impact values
    fig10 = plt.figure("Figure 10")   #figsize with default values
    dftable = pd.DataFrame(index=['Cumulative Return', 'Std. Deviation of Daily Returns', 'Mean of Daily Returns',
                                  'Sharpe Ratio', 'Final Value'],
                           columns=['Impact=0.015','Impact=0.010','Impact=0.005','Impact=0.0025','Impact=0.000'])

    dftable.iloc[0, 0] = f"{cum_ret_SL_015:.6f}"
    dftable.iloc[1, 0] = f"{std_daily_ret_SL_015:.6f}"
    dftable.iloc[2, 0] = f"{avg_daily_ret_SL_015:.6f}"
    dftable.iloc[3, 0] = f"{sharpe_ratio_SL_015:.6f}"
    dftable.iloc[4, 0] = f"{final_val_SL_015:.2f}"

    dftable.iloc[0, 1] = f"{cum_ret_SL_010:.6f}"
    dftable.iloc[1, 1] = f"{std_daily_ret_SL_010:.6f}"
    dftable.iloc[2, 1] = f"{avg_daily_ret_SL_010:.6f}"
    dftable.iloc[3, 1] = f"{sharpe_ratio_SL_010:.6f}"
    dftable.iloc[4, 1] = f"{final_val_SL_010:.2f}"

    dftable.iloc[0, 2] = f"{cum_ret_SL_005:.6f}"
    dftable.iloc[1, 2] = f"{std_daily_ret_SL_005:.6f}"
    dftable.iloc[2, 2] = f"{avg_daily_ret_SL_005:.6f}"
    dftable.iloc[3, 2] = f"{sharpe_ratio_SL_005:.6f}"
    dftable.iloc[4, 2] = f"{final_val_SL_005:.2f}"

    dftable.iloc[0, 3] = f"{cum_ret_SL_0025:.6f}"
    dftable.iloc[1, 3] = f"{std_daily_ret_SL_0025:.6f}"
    dftable.iloc[2, 3] = f"{avg_daily_ret_SL_0025:.6f}"
    dftable.iloc[3, 3] = f"{sharpe_ratio_SL_0025:.6f}"
    dftable.iloc[4, 3] = f"{final_val_SL_0025:.2f}"

    dftable.iloc[0, 4] = f"{cum_ret_SL_000:.6f}"
    dftable.iloc[1, 4] = f"{std_daily_ret_SL_000:.6f}"
    dftable.iloc[2, 4] = f"{avg_daily_ret_SL_000:.6f}"
    dftable.iloc[3, 4] = f"{sharpe_ratio_SL_000:.6f}"
    dftable.iloc[4, 4] = f"{final_val_SL_000:.2f}"


    # Solution from https://stackoverflow.com/questions/35634238/how-to-save-a-pandas-dataframe-table-as-a-png
    fig10, ax10 = plt.subplots(figsize=(12, 2))  # set size frame
    ax10.set_title("In-Sample Q-Learner Metrics for Different Impact Values using 'JPM'", fontsize=12)
    ax10.xaxis.set_visible(False)  # hide the x axis
    ax10.yaxis.set_visible(False)  # hide the y axis
    ax10.set_frame_on(False)  # no visible frame
    summary = pd.plotting.table(ax10, dftable, loc='upper right', colWidths=[0.15] * len(dftable.columns))  # where df is your data frame
    summary.auto_set_font_size(False)  # Activate set fontsize manually
    summary.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
    summary.scale(1.0, 1.0)  # change size table
    # plt.savefig("images/Figure10")
    plt.savefig("images/Table3a")

    # Figure 11: Histogram considering Number of trades

    # Combiine all trades dafatframes and exclude trades of 0 stocks.
    dftrades_all = pd.concat([dftrades_SL_015.loc[(dftrades_SL_015 != 0).any(axis=1)],
                              dftrades_SL_010.loc[(dftrades_SL_010 != 0).any(axis=1)],
                              dftrades_SL_005.loc[(dftrades_SL_005 != 0).any(axis=1)],
                              dftrades_SL_0025.loc[(dftrades_SL_0025 != 0).any(axis=1)],
                              dftrades_SL_000.loc[(dftrades_SL_000 != 0).any(axis=1)] ], axis = 1)
    dftrades_all.columns = range(dftrades_all.columns.size)     # Reset column names
    dftrades_all = dftrades_all.rename(columns={0: "0.015", 1: "0.010", 2: "0.005",
                                                3: "0.0025", 4: "0.000"})
    Number_trades = (~np.isnan(dftrades_all)).sum()

    fig11, ax11 = plt.subplots(figsize=(6, 8))  # set size frame
    Number_trades.plot(kind='bar', color='teal')
    ax11.set_title("Effect of Impact in Number of Trades: In-sample period using 'JPM'", fontsize=12)
    ax11.grid(True, color='gray', linestyle='--', which='major', axis='y')
    ax11.set_xlabel("Impact")
    ax11.set_ylabel("Number of Trades")
    # plt.savefig("images/Figure11")
    plt.savefig("images/Figure7a")



def OutSampleRun():

    ticker = 'JPM'
    start_value = 100000
    ## IN-SAMPLE PERIOD
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    ## OUT-OF-SAMPLE PERIOD
    start_date_out_sample = dt.datetime(2010, 1, 1)
    end_date_out_sample = dt.datetime(2011, 12, 31)    # Last trading day



    # STRATEGY LEARNERS

    # Commission = 0, Impact = 0.015
    learner = sl.StrategyLearner(verbose=False, commission=9.95, impact=0.015)     # constructor
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)  # training phase
    dftrades_SL_015 = learner.testPolicy(symbol=ticker, sd=start_date_out_sample, ed=end_date_out_sample, sv=start_value)  # testing phase
    portvals_SL_015 = mktsim.compute_portvals(orders_info=dftrades_SL_015, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL_015, avg_daily_ret_SL_015, std_daily_ret_SL_015, sharpe_ratio_SL_015, final_val_SL_015 = mktsim.assess_portfolio(
        sd=start_date_out_sample, ed=end_date_out_sample, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL_015)


    # Commission = 0, Impact = 0.010
    learner = sl.StrategyLearner(verbose=False, commission=9.95, impact=0.010)     # constructor
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)  # training phase
    dftrades_SL_010 = learner.testPolicy(symbol=ticker, sd=start_date_out_sample, ed=end_date_out_sample, sv=start_value)  # testing phase
    portvals_SL_010 = mktsim.compute_portvals(orders_info=dftrades_SL_010, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL_010, avg_daily_ret_SL_010, std_daily_ret_SL_010, sharpe_ratio_SL_010, final_val_SL_010 = mktsim.assess_portfolio(
        sd=start_date_out_sample, ed=end_date_out_sample, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL_010)


    # Commission = 0, Impact = 0.005
    learner = sl.StrategyLearner(verbose=False, commission=9.95, impact=0.005)     # constructor
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)  # training phase
    dftrades_SL_005 = learner.testPolicy(symbol=ticker, sd=start_date_out_sample, ed=end_date_out_sample, sv=start_value)  # testing phase
    portvals_SL_005 = mktsim.compute_portvals(orders_info=dftrades_SL_005, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL_005, avg_daily_ret_SL_005, std_daily_ret_SL_005, sharpe_ratio_SL_005, final_val_SL_005 = mktsim.assess_portfolio(
        sd=start_date_out_sample, ed=end_date_out_sample, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL_005)


    # Commission = 0, Impact = 0.0025
    learner = sl.StrategyLearner(verbose=False, commission=9.95, impact=0.0025)     # constructor
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)  # training phase
    dftrades_SL_0025 = learner.testPolicy(symbol=ticker, sd=start_date_out_sample, ed=end_date_out_sample, sv=start_value)  # testing phase
    portvals_SL_0025 = mktsim.compute_portvals(orders_info=dftrades_SL_0025, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL_0025, avg_daily_ret_SL_0025, std_daily_ret_SL_0025, sharpe_ratio_SL_0025, final_val_SL_0025 = mktsim.assess_portfolio(
        sd=start_date_out_sample, ed=end_date_out_sample, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL_0025)


    # Commission = 0, Impact = 0.000
    learner = sl.StrategyLearner(verbose=False, commission=9.95, impact=0.000)     # constructor
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)  # training phase
    dftrades_SL_000 = learner.testPolicy(symbol=ticker, sd=start_date_out_sample, ed=end_date_out_sample, sv=start_value)  # testing phase
    portvals_SL_000 = mktsim.compute_portvals(orders_info=dftrades_SL_000, start_val=start_value,
                                          commission=learner.commission, impact=learner.impact)
    # Get metrics from TOS portfolio values
    cum_ret_SL_000, avg_daily_ret_SL_000, std_daily_ret_SL_000, sharpe_ratio_SL_000, final_val_SL_000 = mktsim.assess_portfolio(
        sd=start_date_out_sample, ed=end_date_out_sample, syms=[], allocs=[], sv=1, gen_plot=False, dfvalue=portvals_SL_000)




    dfcompare =  pd.concat([portvals_SL_015, portvals_SL_010, portvals_SL_005, portvals_SL_0025, portvals_SL_000], axis = 1)
    dfcompare = dfcompare.rename(columns={0: "Impact=0.015", 1: "Impact=0.010", 2: "Impact=0.005", 3: "Impact=0.0025",
                                          4: "Impact=0.000"})
    dfcompare.fillna(method="ffill", inplace=True)   # fill forward first
    dfcompare.fillna(method="bfill", inplace=True)  # fill backwards second
    dfcompare = dfcompare/dfcompare.iloc[0]     # Normalized portfolio values

    # Figure 12: Plot chart of Out-of-sample Q Learner portfolio considering different impact
    fig12 = plt.figure("Figure 12", figsize=(7,4.8))   #figsize with default values
    ax = dfcompare["Impact=0.015"].plot(color="teal")
    ax = dfcompare["Impact=0.010"].plot(color="blueviolet")
    ax = dfcompare["Impact=0.005"].plot(color="darkorange")
    ax = dfcompare["Impact=0.0025"].plot(color="limegreen")
    ax = dfcompare["Impact=0.000"].plot(color="red")
    # ax = dfcompare["TOS"].plot(color="gray")
    ax.set_title("Effect of Impact for Out-of-Sample Period in Q-Learner using 'JPM'", fontsize=12)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized portfolio value")
    ax.set_xticks(pd.date_range(start_date_out_sample, end_date_out_sample, periods=12))

    # plt.savefig("images/Figure12")
    plt.savefig("images/Figure6")




    # Figure 13: Table showing to 6 digits the metrics of Strategy Learner with different impact values
    fig13 = plt.figure("Figure 13")   #figsize with default values
    dftable = pd.DataFrame(index=['Cumulative Return', 'Std. Deviation of Daily Returns', 'Mean of Daily Returns',
                                  'Sharpe Ratio', 'Final Value'],
                           columns=['Impact=0.015','Impact=0.010','Impact=0.005','Impact=0.0025','Impact=0.000'])

    dftable.iloc[0, 0] = f"{cum_ret_SL_015:.6f}"
    dftable.iloc[1, 0] = f"{std_daily_ret_SL_015:.6f}"
    dftable.iloc[2, 0] = f"{avg_daily_ret_SL_015:.6f}"
    dftable.iloc[3, 0] = f"{sharpe_ratio_SL_015:.6f}"
    dftable.iloc[4, 0] = f"{final_val_SL_015:.2f}"

    dftable.iloc[0, 1] = f"{cum_ret_SL_010:.6f}"
    dftable.iloc[1, 1] = f"{std_daily_ret_SL_010:.6f}"
    dftable.iloc[2, 1] = f"{avg_daily_ret_SL_010:.6f}"
    dftable.iloc[3, 1] = f"{sharpe_ratio_SL_010:.6f}"
    dftable.iloc[4, 1] = f"{final_val_SL_010:.2f}"

    dftable.iloc[0, 2] = f"{cum_ret_SL_005:.6f}"
    dftable.iloc[1, 2] = f"{std_daily_ret_SL_005:.6f}"
    dftable.iloc[2, 2] = f"{avg_daily_ret_SL_005:.6f}"
    dftable.iloc[3, 2] = f"{sharpe_ratio_SL_005:.6f}"
    dftable.iloc[4, 2] = f"{final_val_SL_005:.2f}"

    dftable.iloc[0, 3] = f"{cum_ret_SL_0025:.6f}"
    dftable.iloc[1, 3] = f"{std_daily_ret_SL_0025:.6f}"
    dftable.iloc[2, 3] = f"{avg_daily_ret_SL_0025:.6f}"
    dftable.iloc[3, 3] = f"{sharpe_ratio_SL_0025:.6f}"
    dftable.iloc[4, 3] = f"{final_val_SL_0025:.2f}"

    dftable.iloc[0, 4] = f"{cum_ret_SL_000:.6f}"
    dftable.iloc[1, 4] = f"{std_daily_ret_SL_000:.6f}"
    dftable.iloc[2, 4] = f"{avg_daily_ret_SL_000:.6f}"
    dftable.iloc[3, 4] = f"{sharpe_ratio_SL_000:.6f}"
    dftable.iloc[4, 4] = f"{final_val_SL_000:.2f}"


    # Solution from https://stackoverflow.com/questions/35634238/how-to-save-a-pandas-dataframe-table-as-a-png
    fig13, ax13 = plt.subplots(figsize=(12, 2))  # set size frame
    ax13.set_title("Out-of-Sample Q-Learner Metrics for Different Impact Values using 'JPM'", fontsize=12)
    ax13.xaxis.set_visible(False)  # hide the x axis
    ax13.yaxis.set_visible(False)  # hide the y axis
    ax13.set_frame_on(False)  # no visible frame
    summary = pd.plotting.table(ax13, dftable, loc='upper right', colWidths=[0.15] * len(dftable.columns))  # where df is your data frame
    summary.auto_set_font_size(False)  # Activate set fontsize manually
    summary.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
    summary.scale(1.0, 1.0)  # change size table
    # plt.savefig("images/Figure13")
    plt.savefig("images/Table3b")


    # Figure 14: Histogram considering Number of trades

    # Combiine all trades dafatframes and exclude trades of 0 stocks.
    dftrades_all = pd.concat([dftrades_SL_015.loc[(dftrades_SL_015 != 0).any(axis=1)],
                              dftrades_SL_010.loc[(dftrades_SL_010 != 0).any(axis=1)],
                              dftrades_SL_005.loc[(dftrades_SL_005 != 0).any(axis=1)],
                              dftrades_SL_0025.loc[(dftrades_SL_0025 != 0).any(axis=1)],
                              dftrades_SL_000.loc[(dftrades_SL_000 != 0).any(axis=1)] ], axis = 1)
    dftrades_all.columns = range(dftrades_all.columns.size)     # Reset column names
    dftrades_all = dftrades_all.rename(columns={0: "0.015", 1: "0.010", 2: "0.005",
                                                3: "0.0025", 4: "0.000"})
    Number_trades = (~np.isnan(dftrades_all)).sum()

    fig14, ax14 = plt.subplots(figsize=(6, 8))  # set size frame
    Number_trades.plot(kind='bar', color='teal')
    ax14.set_title("Effect of Impact in Number of Trades: Out-of-sample period using 'JPM'", fontsize=12)
    ax14.grid(True, color='gray', linestyle='--', which='major', axis='y')
    ax14.set_xlabel("Impact")
    ax14.set_ylabel("Number of Trades")
    # plt.savefig("images/Figure14")
    plt.savefig("images/Figure7b")


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
