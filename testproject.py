""""""
""" testproject.py	  	   		  	  		  		  		    	 		 		   		 		  

Code initializing/running all necessary files for the report

Student Name: Jorge Salvador Aguilar Moreno  		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jmoreno62	  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845924		  	   		  	  		  		  		    	 		 		   		 		  
"""

import math
import sys
import pandas as pd
import numpy as np
import random as rand
import datetime as dt
import matplotlib.pyplot as plt
import marketsimcode as mktsim
from indicators import indicators

import ManualStrategy as ms
import experiment1 as exp1
import experiment2 as exp2
import StrategyLearner as sl



def Project8():

    # Random seed
    # rand.seed(903845924)

    ### 3.3.1 Manual Rule-Based Trader
    # In-sample period
    ms.InSampleRun()
    # Out-of-sample period
    ms.OutSampleRun()

    # 3.3.2 Strategy Learner: Reinforcement-based Learner


    # 3.3.3 Experiment 1
    # In-sample period
    exp1.InSampleRun()
    # Out-of-sample period
    exp1.OutSampleRun()

    # 3.3.4 Experiment 2
    # In-sample period
    exp2.InSampleRun()
    # Out-of-sample period
    exp2.OutSampleRun()


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jmoreno62"


if __name__ == "__main__":
    Project8()
