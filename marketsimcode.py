""""""
"""MC2-P1: Market simulator.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
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
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt


def compute_portvals(
        orders_info="./orders/orders.csv",
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param orders_info: 3 options: 1)Path of the order file, 2) the file object, or 3) a pandas DataFrame
    :type orders_info: str, file object, or DataFrame
    :param start_val: The starting value of the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		  	  		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		  	  		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		  	  		  		  		    	 		 		   		 		  
    """
    # this is the function the autograder will call to test your code  		  	   		  	  		  		  		    	 		 		   		 		  
    # NOTE: orders_info may be a string, or it may be a file object. Your
    # code should work correctly with either input  		  	   		  	  		  		  		    	 		 		   		 		  
    # Your code here (it addresses the string case with isinstance())
    # Drop days not trade with .dropna
    # Fill upwards and backwards

    # JM
    # dataframe orders
    # This represents sorted data of orders to be applied to stock
    if isinstance(orders_info, str): #  Check if 'order_file' is a string
        orders = pd.read_csv(orders_info)  # read csv file

    if isinstance(orders_info, pd.DataFrame): #  Check if 'order_file' is a pandas DataFrame
        orders = pd.DataFrame(columns=['Date','Symbol', 'Order', 'Shares'])
        orders.loc[:,'Date'] = orders_info.index.strftime("%Y-%m-%d")
        orders.loc[:, 'Symbol'] = orders_info.columns[0]
        # Create 'temp' Series ofr 'orders_info' without indices and column name. We need this to copy it into 'orders' column
        if orders_info.size>1:  #If 'orders_info' DataFrame has more than 1 entry, we need to reset index
            temp = orders_info.squeeze().reset_index(drop=True)     # .squeeze() function changes 1-col DataFrame to Series
        else:
            temp = orders_info.squeeze()    # When DataFrame has only 1 entry, result is a scalar
        orders['Shares'] = temp     # This is the same as orders.loc[:,'Shares'] = temp
        orders.loc[orders['Shares']<0,'Order']  = 'SELL'    # If 'Shares' is negative, 'Order' column is 'SELL'
        orders.loc[orders['Shares']>=0, 'Order'] = 'BUY'  # If 'Shares' is positive, 'Order' column is 'BUY'
        orders = orders.dropna()    #drop rows with 'nan' in 'Order' column
        orders['Shares'] = orders['Shares'].abs()


    else:   # if not, it is assumed it is a file object (TextIOWrapper)
        text = orders_info.read()
        text = text.split() # Create lists of each text line
        orders = pd.DataFrame(columns=text[0].split(","))
        for i in range(1,len(text)):
            orders.loc[len(orders.index)] = text[i].split(",")
        orders.loc[:,'Shares'] = pd.to_numeric(orders.loc[:,'Shares'])

    orders = orders.sort_values(by=['Date'])  # orders in order by Date
    orders = orders.reset_index(drop=True)  # reset index, drop index to avoid inserting new column with old indices
    # Update 'orders' to capture condition when order is on weekend
    start_date = orders['Date'].iloc[0]
    end_date = orders['Date'].iloc[-1]
    SPY = get_data([], pd.date_range(start_date, end_date),addSPY=True) # Get trading days from SPY

    drop_rows = []
    for i in range(len(orders)):
        date = orders.iloc[i]['Date']
        date = dt.datetime.strptime(date, "%Y-%m-%d")  # Convert string to datetime object
        # Edge case of receiving order on a non-trading day
        if date not in SPY.index:
            drop_rows.append(i)
    orders = orders.drop(drop_rows)
    orders = orders.reset_index(drop=True)  # reset index, drop index to avoid inserting new column with old indices

        # The following inactive code assumes the trade is executed in the following trading date, but it's not req'd
        # if date not in SPY.index:
            # while date not in SPY.index:
            #     date = date + dt.timedelta(days=1)
            # date = date.strftime("%Y-%m-%d")  # Convert datetime object back to string
            # orders.loc[i,'Date'] = date

    # dataframe dfprices
    # This represents prices of each stock included in 'orders' in trading days
    start_date = orders['Date'].iloc[0] # Update start_date
    end_date = orders['Date'].iloc[-1]  # Update end_date
    symbols = orders['Symbol'].unique()  # identify unique symbols
    dfprices = get_data(symbols, pd.date_range(start_date, end_date),
                        addSPY=True)  # Get data including SPY to bring trading days only
    dfprices['CASH'] = 1.  # add column with 1s in dataframe
    if 'SPY' not in symbols:  # Once we have the trading days, drop SPY if it is not in portfolio
        dfprices = dfprices.drop(columns=['SPY'])
    dfprices.fillna(method="ffill", inplace=True)   # fill forward first
    dfprices.fillna(method="bfill", inplace=True)  # fill backwards second

    # dataframe dftrades
    # This represents changes in number of shares on particular days on my assets. It includes stock and cash
    dftrades = pd.DataFrame(index=dfprices.index,
                            columns=dfprices.columns)  # Create empty df with same index and columns as dfprices
    dftrades.iloc[:] = 0.  # Fill all the rows with Os. Remember mutability of dataframes

    for i in range(len(orders)):
        # Calculate total cost of transaction
        date = orders.iloc[i]['Date']
        symbol = orders.iloc[i]['Symbol']
        num_shares = orders.iloc[i]['Shares']
        cost = dfprices.loc[date, symbol]
        total_cost = num_shares * cost

        if orders.iloc[i]['Order'] == "BUY":
            # Log "buy" information in dftrades
            dftrades.loc[date, symbol] += num_shares
            dftrades.loc[date, 'CASH'] -= total_cost
            if num_shares != 0:     # Catch 'BUY's of 0 shares
                dftrades.loc[date, 'CASH'] -= commission + impact*total_cost  # commission and impact as deduction of cash

        elif orders.iloc[i]['Order'] == "SELL":
            dftrades.loc[date, symbol] -= num_shares
            dftrades.loc[date, 'CASH'] += total_cost
            dftrades.loc[date, 'CASH'] -= commission + impact*total_cost  # commission and impact as deduction of cash

    # dataframe dfholdings
    # This represents on each particular day, how much of each asset I am holding. It includes stock and cash
    dfholdings = pd.DataFrame(index=dfprices.index,
                              columns=dfprices.columns)  # Create empty df with same index and columns as dfprices
    dfholdings.iloc[:] = 0.  # Fill all the rows with Os. Remember mutability of dataframes
    dfholdings.iloc[:, :] = dftrades.iloc[:, :]  # Copy information from dftrades. This approach is used due to mutability
    dfholdings.loc[start_date, 'CASH'] += start_val  # For 'CASH' column, add start_val in first row of 'dftrades'
    dfholdings = dfholdings.cumsum()  # cumulative sum of transactions

    # dataframe dfvalue
    # This represents what is the monetary ($) value of each asset for each day
    dfvalue = pd.DataFrame(index=dfprices.index,
                           columns=dfprices.columns)  # Create empty df with same index and columns as dfprices
    dfvalue.iloc[:] = 0.  # Fill all the rows with Os. Remember mutability of dataframes
    dfvalue = dfprices * dfholdings

    # dataframe portvals
    # This represents the sum of each row of dfvalue
    portvals = pd.DataFrame(index=dfprices.index, columns=['Port_Vals'])
    portvals = dfvalue.sum(axis=1)

    # dataframe dfleverage is not required for this assignment

    return portvals


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jmoreno62"


def assess_portfolio(
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        syms=["GOOG", "AAPL", "GLD", "XOM"],
        allocs=[0.1, 0.2, 0.3, 0.4],
        sv=1000000,
        rfr=0.0,
        sf=252.0,
        gen_plot=False,
        dfvalue = pd.DataFrame(),
):
    """
    Estimate a set of test points given the model we built.

    :param sd: A datetime object that represents the start date, defaults to 1/1/2008
    :type sd: datetime
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009
    :type ed: datetime
    :param syms: A list of 2 or more symbols that make up the portfolio (note that your code should support any symbol in the data directory)
    :type syms: list
    :param allocs:  A list of 2 or more allocations to the stocks, must sum to 1.0
    :type allocs: list
    :param sv: The starting value of the portfolio
    :type sv: int
    :param rfr: The risk free return per sample period that does not change for the entire date range (a single number, not an array)
    :type rfr: float
    :param sf: Sampling frequency per year
    :type sf: float
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your
        code with gen_plot = False.
    :type gen_plot: bool
    :return: A tuple containing the cumulative return, average daily returns,
        standard deviation of daily returns, Sharpe ratio and end value
    :rtype: tuple
    """

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)

    if len(syms)>1:
        prices_all = get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

        # Get daily portfolio value
        # port_val = prices_SPY  # add code here to compute daily portfolio values
        normed = prices / prices.iloc[0, :]  # normalized prices to the first day, first row is 1.0 for all
        alloced = normed * allocs  # mulitply normed series with allocations
        pos_vals = alloced * sv  # multiply by initial investment, i.e. sv
        port_val = pos_vals.sum(axis=1)  # sum each row to get value of portfolio each day
    else:
        if syms == []:   # Get metrics for portfolio
            port_val = dfvalue
        elif syms[0] == "$SPX":  # Get metrics for $SPX
            port_val = get_data(syms, pd.date_range(sd, ed), addSPY=True)
            port_val = port_val.drop(columns=['SPY'])   #Add SPY to drop dates didn't trade, and then drop SPY
        else:
            print("Argument for syms not valid. Leave blank to assess portfolio")

    # Get daily returns
    daily_ret = (port_val / port_val.shift(1)) - 1
    daily_ret.iloc[0] = 0

    # Get portfolio statistics (note: std_daily_ret = volatility)
    # cr: cumulative return
    # adr: average daily returns
    # sddr: st. dev. of daily returns or volatility
    # sr: Sharpe ratio
    cr, adr, sddr, sr = [
        (port_val.iloc[-1] / port_val.iloc[0]) - 1,
        daily_ret[1:].mean(),
        daily_ret[1:].std(),
        (sf) ** (1 / 2) * (daily_ret[1:].mean() - rfr) / (daily_ret[1:].std()),
    ]  # add code here to compute stats

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat(
            [port_val / port_val.iloc[0], prices_SPY / prices_SPY.iloc[0]], keys=["Portfolio", "SPY"], axis=1
        )
        ax = df_temp.plot(title="Portfolio vs SP500", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        plt.show()
    else:
        pass

        # Add code here to properly compute end value
    ev = port_val.iloc[-1]

    return cr, adr, sddr, sr, ev

def test_code():
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		  	  		  		  		    	 		 		   		 		  
    """
    # this is a helper function you can use to test your code  		  	   		  	  		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		  	  		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		  	  		  		  		    	 		 		   		 		  

    of = "./orders/orders2.csv"
    sv = 1000000

    # Test considering a file object
    # file = open(of)
    # of = file

    # Process orders
    portvals = compute_portvals(orders_info=of, start_val=sv)

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[
            portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    orders = pd.read_csv(of)  # read csv file
    orders = orders.sort_values(by=['Date'])  # orders in order by Date
    orders = orders.reset_index(drop=True)  # reset index, drop index to avoid inserting new column with old indices
    start_date = orders['Date'].iloc[0]
    end_date = orders['Date'].iloc[-1]

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio, final_val = assess_portfolio(
        sd=start_date,
        ed=end_date,
        syms=[],
        allocs=[],
        sv=1,
        gen_plot=False,
        dfvalue=portvals,
        )

    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY,final_val_SPY = assess_portfolio(
        sd=start_date,
        ed=end_date,
        syms=['$SPX'],
        allocs=[],
        sv=1,
        gen_plot=False)

    # Compare portfolio against $SPX  		  	   		  	  		  		  		    	 		 		   		 		  
    # print(f"Date Range: {start_date} to {end_date}")
    # print()
    # print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    # print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    # print()
    # print(f"Cumulative Return of Fund: {cum_ret}")
    # print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    # print()
    # print(f"Standard Deviation of Fund: {std_daily_ret}")
    # print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    # print()
    # print(f"Average Daily Return of Fund: {avg_daily_ret}")
    # print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    # print()
    # print(f"Final Portfolio Value: {final_val}")
    # print(f"Final $SPX Value: {final_val_SPY}")


if __name__ == "__main__":
    test_code()
