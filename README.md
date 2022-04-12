# UChicagoTradingCompetition2022
3rd place Team's Winning Trading Bots and Portfolio Allocation method for the 2022 UChicago Trading Competition

Case Objectives: 2022 UChicago Trading Competition Participant Case Packet.pdf

Case 1 (Market Making) and Case 2 (Options Trading) Trading Bots:
  - Navigate inside utc_xchange_v2.0 folder
  - Follow the README.md inside that folder to run case1bot.py and case2bot.py on the exchange

Case 3 Portfolio Allocation Method:
  - Navigate inside Case3 folder
  - allocate.py takes a single row of actual_prices.csv (actual price for each of the 9 assetts at a single point in time), analyst_1_prediction.csv (predicted price for each of the 9 assetts for the next point in time from analyst 2), analyst_2_prediction.csv, and analyst_3_prediction.csv as input, and returns optimal portfolio weights for each of the 9 assetts for the next point in time
