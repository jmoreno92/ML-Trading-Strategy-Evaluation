# Trading-Strategy-Evaluation
A trading strategy is a structured methodology to exchange securities in the eq- uity market. Nowadays, with the large amount of markets data and computa- tional power developments, about 60-75% of the overall trading volume is performed through a pre-programmed strategy.
To develop a successful strategy, it is required to have a solid understanding of technical indicators, trading actions, and investing. Moreover, this knowledge can be leveraged with machine learning. This project integrates these concepts to implement two trading strategies:
 - Manual rule-based strategy: predefined rules using technical indicators to enter and exit stock positions.
 - Strategy learner: trading policy based on a reinforcement-based learner us- ing the same indicators.
Once implemented, the strategies are compared considering the following rules:
 - Trades only using the symbol ‘JPM’ for J.P. Morgan Chase & Co.
 - In-sample period from January 1, 2008 to December 31, 2009.
 - Out-of-sample period from January 1, 2010 to December 31, 2011.
 - Starting cash of $100,000, where the only allowable positions are 1,000 shares long, 1,000 shares short, or 0 shares. Only buy/sell actions are allowed.
 - Transaction cots of $9.95 for commission and 0.005 for impact, except for ex-
periment 2 (Refer to section 6 of the report)
The strategies are also compared to a benchmark of buying 1,000 shares of ‘JPM’.
