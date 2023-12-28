""""""
""" TheoreticallyOptimalStrategy.py	  	   		  	  		  		  		    	 		 		   		 		  

Code implementing a TheoreticallyOptimalStrategy (details below). It should implement testPolicy(), which 
returns a trades data frame (see below).

Student Name: Jorge Salvador Aguilar Moreno  		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jmoreno62	  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845924		  	   		  	  		  		  		    	 		 		   		 		  
"""

import math
import sys
import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data



def testPolicy(symbol="AAPL",
               sd=dt.datetime(2010, 1, 1),
               ed=dt.datetime(2011,12,31),
               sv = 100000):
    stock_price = get_data([symbol], pd.date_range(sd,ed),addSPY=True) # Get trading days from SPY

    if 'SPY' != symbol:  # Once we have the trading days, drop SPY if it is not in portfolio
        stock_price = stock_price.drop(columns=['SPY'])
    stock_price.fillna(method="ffill", inplace=True)   # fill forward first
    stock_price.fillna(method="bfill", inplace=True)  # fill backwards second
    # plot_data(stock_price)

    # Create 'dftrades' DataFrame with the same time index as the stock_price
    dftrades = pd.DataFrame(index=stock_price.index, columns=[symbol])
    dftrades.iloc[:]=0.   # Fill all the rows with Os. Remember mutability of dataframes

    position = 0  # Initial position. Allowable positions are 1000 shares long, 1000 shares short, and 0 shares

    for i in range(len(dftrades) - 1):

        if stock_price.iloc[i,0] < stock_price.iloc[i+1,0]: # Price will go up. Go long
            trade = min(2000, 1000 - position)  # maximum trade using min function
            position = 1000

        elif stock_price.iloc[i,0] == stock_price.iloc[i+1,0]:  # Price will be the same. Do nothing
            trade = 0

        elif stock_price.iloc[i,0] > stock_price.iloc[i+1,0]:   # Price will go down. Go short
            trade = max(-2000, -1000 - position)  # minimum trade using max function
            position = -1000

        else:
            pass
        dftrades.iloc[i] = trade

    return dftrades



def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jmoreno62"


if __name__ == "__main__":
    testPolicy()