""""""
""" StrategyLearner.py	  	   		  	  		  		  		    	 		 		   		 		  

Code implementing a StrategyLearner object (your ML strategy).
It should implement testPolicy(), which returns a trades data frame. The main part of this code should 
call marketsimcode as necessary to generate the plotes used in the report.

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
import TheoreticallyOptimalStrategy as tos
import marketsimcode as mktsim
from indicators import indicators


# TODO: implement testPolicy()

def test_code():


    #Parameters for the Project
    ticker = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009,12,31)
    start_value = 100000
    lookback = 20

    # Call indicators
    JPM = indicators(symbol=ticker, sd=start_date, ed=end_date, sv=start_value)

    # FIGURE 1 AND 2
    JPM.run()





def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jmoreno62"


if __name__ == "__main__":
    # TODO: Main part of this could should call marketsimcode to generate plots of this report
    test_code()
