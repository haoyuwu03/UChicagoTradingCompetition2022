from dataclasses import astuple
from datetime import datetime
from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto

import asyncio
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm

# Setting available strike prices
option_strikes = [90, 95, 100, 105, 110]

class Case2ExampleBot(UTCBot):
    """
    An example bot for Case 2 of the 2021 UChicago Trading Competition. We recommend that you start
    by reading through this bot and understanding how it works. Then, make a copy of this file and
    start trying to write your own bot!
    """

    async def handle_round_started(self):
        """
        This function is called when the round is started. You should do your setup here, and
        start any tasks that should be running for the rest of the round.
        """
        # play around with these parameters:
        # 3
        self.edge_buy = 2
        self.edge_sell = 2
        # 15
        self.buy_option_q = 15
        self.sell_option_q = 15
        self.increment = 0.2
        self.position_limit = 100
        self.update_count = 0
        self.skip_updates = 2
        self.num_lots = 0
        self.lot_expand = 3
        # This variable will be a map from asset names to positions. We start out by initializing it
        # to zero for every asset.
        self.positions = {}
        self.order_ids = {}
        self.asset_names = []
        #self.fills = {}
        # For storing market orders
        self.positions["UC"] = 0

        # For storing options limit orders
        for strike in option_strikes:
            for flag in ["C", "P"]:
                asset_name = f"UC{strike}{flag}"
                self.asset_names.append(asset_name)
                self.positions[asset_name] = 0
                # id for bid, ask
                self.order_ids[asset_name] = ["",""]
                for i in range(self.num_lots):
                    self.asset_names.append(asset_name+str(i))
                    self.positions[asset_name+str(i)] = 0
                    self.order_ids[asset_name+str(i)] = ["",""]
                
            

        # Stores the current day (starting from 0 and ending at 5). This is a floating point number,
        # meaning that it includes information about partial days
        self.current_day = 0
        self.time_to_expiry = (26-self.current_day) / 252

        # Stores the current value of the underlying asset
        self.previous_price = 100
        self.underlying_price = 100
        self.old_vol = 0.0008269438519741517

        # Store risk measures
        self.risks = {"d": 0, "g": 0, "t":0, "v":0}
        self.lowest_ask = 0
        self.lowest_bid = 0
        self.place_ioc_orders()
        #self.place_straddle_order()

    def reset_edge(self):
        self.edge_buy = 5
        self.edge_sell = 5

    async def place_straddle_order(self):
        requests = []
        for strike in option_strikes:
            asset_name = f"UC{strike}P"
            requests.append(
                self.place_order(
                                asset_name,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.BID,
                                1,
                                self.highest_ask,
                            )
                        )
            asset_name = f"UC{strike}C"
            requests.append(
                self.place_order(
                                asset_name,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.BID,
                                1,
                                self.lowest_bid,
                            )
                        )
        responses = await asyncio.gather(*requests)
        for resp in responses:
            assert resp.ok
    async def place_ioc_orders(self):
        requests = []
        requests.append(
            self.place_order(
                "UC",
                pb.OrderSpecType.IOC,
                pb.OrderSpecSide.ASK,
                15,
                130
            )
        )
        
        requests.append(
            self.place_order(
                "UC",
                pb.OrderSpecType.IOC,
                pb.OrderSpecSide.BID,
                15,
                90
            )
        )
        responses = await asyncio.gather(*requests)
        for resp in responses:
            assert resp.ok
        
    def update_EMA(self,old_EMA, new_data):
        return new_data * (2/(200)) + old_EMA * (1-(2/(200)))

    def update_std(self,old_std,new_ret):
        old_avg = old_std**2
        new_data = new_ret**2
        return (self.update_EMA(old_avg,new_data))**0.5

    def compute_vol_estimate(self) -> float:
        """
        This function is used to provide an estimate of underlying's volatility. Because this is
        an example bot, we just use a placeholder value here. We recommend that you look into
        different ways of finding what the true volatility of the underlying is.
        """
        new_vol = self.update_std(self.old_vol,np.log(self.underlying_price/self.previous_price))
        self.old_vol = new_vol
        annual_vol = (new_vol * (200)**0.5) * (252)**0.5
        return annual_vol
    
    # Black Scholes
    
    def d1(self,S,K,T,r,sigma):
        return(np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))

    def d2(self,S,K,T,r,sigma):
        return self.d1(S,K,T,r,sigma)-sigma*np.sqrt(T)

    def bs_call(self,S,K,T,r,sigma):
        return S*norm.cdf(self.d1(S,K,T,r,sigma))-K*np.exp(-r*T)*norm.cdf(self.d2(S,K,T,r,sigma))

    def bs_put(self,S,K,T,r,sigma):
        return K*np.exp(-r*T)-S+self.bs_call(S,K,T,r,sigma)

    # Implied Volatility:

    def iv_call(self,S,K,T,r,C):
        return max(0, fsolve((lambda sigma: np.abs(self.bs_call(S,K,T,r,sigma) - C)), [1])[0])
                        
    def iv_put(self,S,K,T,r,P):
        return max(0, fsolve((lambda sigma: np.abs(self.bs_put(S,K,T,r,sigma) - P)), [1])[0])
    
    # Greeks

    def delta_call(self,S,K,T,C):
        sigma = self.iv_call(S,K,T,0,C)
        return 100 * norm.cdf(self.d1(S,K,T,0,sigma))

    def gamma_call(self,S,K,T,C):
        sigma = self.iv_call(S,K,T,0,C)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma))/(S * sigma * np.sqrt(T))

    def vega_call(self,S,K,T,C):
        sigma = self.iv_call(S,K,T,0,C)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma)) * S * np.sqrt(T)

    def theta_call(self,S,K,T,C):
        sigma = self.iv_call(S,K,T,0,C)
        return 100 * S * norm.pdf(self.d1(S,K,T,0,sigma)) * sigma/(2 * np.sqrt(T))

    def delta_put(self,S,K,T,C):
        sigma = self.iv_put(S,K,T,0,C)
        return 100 * (norm.cdf(self.d1(S,K,T,0,sigma)) - 1)

    def gamma_put(self,S,K,T,C):
        sigma = self.iv_put(S,K,T,0,C)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma))/(S * sigma * np.sqrt(T))

    def vega_put(self,S,K,T,C):
        sigma = self.iv_put(S,K,T,0,C)
        return 100 * norm.pdf(self.d1(S,K,T,0,sigma)) * S * np.sqrt(T)

    def theta_put(self,S,K,T,C):
        sigma = self.iv_put(S,K,T,0,C)
        return 100 * S * norm.pdf(self.d1(S,K,T,0,sigma)) * sigma/(2 * np.sqrt(T))

    def check_valid_trade(self, sell_or_buy, tag,S,K,T,C,q):
        if tag == "C":
            d = self.delta_call(S,K,T,C) * sell_or_buy * q
            g = self.gamma_call(S,K,T,C) * sell_or_buy * q
            t = self.theta_call(S,K,T,C) * sell_or_buy * q
            v = self.vega_call(S,K,T,C) * sell_or_buy * q
        else:
            d = self.delta_put(S,K,T,C) * sell_or_buy * q
            g = self.gamma_put(S,K,T,C) * sell_or_buy * q
            t = self.theta_put(S,K,T,C) * sell_or_buy * q
            v = self.vega_put(S,K,T,C) * sell_or_buy * q
        return [abs(d) < 2000 and abs(g) < 5000 and abs(t) < 1000000 and abs(v) < 500000, d, g, t, v]

    def check_cum_limits(self, d,g,t,v):
        d = self.risks["d"] + d
        g = self.risks["g"] + g
        t = self.risks["t"] + t
        v = self.risks["v"] + v
        return {"d":[abs(d) > 2000,d], "g":[abs(g) > 5000,g], "t":[abs(t) > 1000000,t], "v":[abs(v) > 500000,v]}

    def compute_options_price(
        self,
        flag: str,
        underlying_px: float,
        strike_px: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:
        """
        This function should compute the price of an option given the provided parameters. Some
        important questions you may want to think about are:
            - What are the units associated with each of these quantities?
            - What formula should you use to compute the price of the option?
            - Are there tricks you can use to do this more quickly?
        You may want to look into the py_vollib library, which is installed by default in your
        virtual environment.
        """
        if flag=="C":
            price = self.bs_call(underlying_px,strike_px,time_to_expiry,0,volatility)
        else:
            price = self.bs_put(underlying_px,strike_px,time_to_expiry,0,volatility)
        return price
    async def update_current_risks(self):
        # keep track of portfolio of fills and recalculate risk for each position
        pass
    async def update_options_quotes(self):
        """
        This function will update the quotes that the bot has currently put into the market.

        In this example bot, the bot won't bother pulling old quotes, and will instead just set new
        quotes at the new theoretical price every time a price update happens. We don't recommend
        that you do this in the actual competition
        """
        
        vol = self.compute_vol_estimate()

        requests = []
        orders = []
        for strike in option_strikes:
            flag = "C"
            # Price a Call
            self.reset_edge()
            asset_name = f"UC{strike}{flag}"
            theo = self.compute_options_price(
                    flag, self.underlying_price, strike, self.time_to_expiry, vol
            )
            self.edge_buy = self.edge_buy * 1+(self.positions[asset_name]/self.position_limit) 
            self.edge_sell = self.edge_sell * 1-(self.positions[asset_name]/self.position_limit)
            iv_e_buy = self.iv_call(self.underlying_price, strike, self.time_to_expiry, 0, self.edge_buy)
            iv_e_sell = self.iv_call(self.underlying_price, strike, self.time_to_expiry, 0, self.edge_sell)


            res_buy = self.check_valid_trade(1,flag,self.underlying_price, strike, self.time_to_expiry, theo-iv_e_buy,self.buy_option_q)
            res_sell = self.check_valid_trade(-1,flag,self.underlying_price, strike, self.time_to_expiry, theo+iv_e_sell,self.sell_option_q)
            while not res_buy[0]:
                self.edge_buy += self.increment
                iv_e_buy = self.iv_call(self.underlying_price, strike, self.time_to_expiry, 0, self.edge_buy)
            while not res_sell[0]:
                self.edge_sell += self.increment
                iv_e_sell = self.iv_call(self.underlying_price, strike, self.time_to_expiry, 0, self.edge_sell)
            
            limits_results_buy = self.check_cum_limits(res_buy[1],res_buy[2],res_buy[3],res_buy[4])
            limits_results_sell = self.check_cum_limits(res_sell[1],res_sell[2],res_sell[3],res_sell[4])
                
            bid_price = round(theo - iv_e_buy, 1)
            ask_price = round(theo + iv_e_sell, 1)

            
            # If can't buy then sell
            if limits_results_buy["g"][0] or limits_results_buy["v"][0] or limits_results_buy["t"][0]:
                # sell
                if self.positions[asset_name] > -self.position_limit:
                    requests.append(
                        self.modify_order(
                            self.order_ids[asset_name][1],
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.ASK,
                            self.sell_option_q,
                            ask_price,
                        )
                    )
                    orders.append([1,asset_name])

                    for i in range(self.num_lots):
                        requests.append(
                            self.modify_order(
                                self.order_ids[asset_name+str(i)][1],
                                asset_name,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.ASK,
                                self.sell_option_q,
                                ask_price+self.lot_expand*(i+1),
                            )
                        ) 
                        orders.append([1,asset_name+str(i)])
                    
                    
                
            # If can't sell then buy
            elif limits_results_sell["g"][0] or limits_results_sell["v"][0] or limits_results_sell["t"][0]:
                # buy   
                if self.positions[asset_name] < self.position_limit:
                    requests.append(
                        self.modify_order(
                            self.order_ids[asset_name][0],
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.BID,
                            self.buy_option_q,  # How should this quantity be chosen?
                            bid_price,  # How should this price be chosen?
                        )
                    )
                    orders.append([0,asset_name])
                    for i in range(self.num_lots):
                        requests.append(
                            self.modify_order(
                                self.order_ids[asset_name+str(i)][0],
                                asset_name,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.BID,
                                self.buy_option_q,
                                bid_price-self.lot_expand*(i+1),
                            )
                        ) 
                        orders.append([0,asset_name+str(i)])
            # otherwise do both because it doesn't matter
            else:
                if self.positions[asset_name] < self.position_limit:
                    requests.append(
                        self.modify_order(
                            self.order_ids[asset_name][0],
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.BID,
                            self.buy_option_q,  # How should this quantity be chosen?
                            bid_price,  # How should this price be chosen?
                        )
                    )
                    orders.append([0,asset_name])
                    for i in range(self.num_lots):
                        requests.append(
                            self.modify_order(
                                self.order_ids[asset_name+str(i)][0],
                                asset_name,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.BID,
                                self.buy_option_q,
                                bid_price-self.lot_expand*(i+1),
                            )
                        ) 
                        orders.append([0,asset_name+str(i)])
                if self.positions[asset_name] > -self.position_limit:
                    requests.append(
                        self.modify_order(
                            self.order_ids[asset_name][1],
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.ASK,
                            self.sell_option_q,
                            ask_price,
                        )
                    )
                    orders.append([1,asset_name])
                    for i in range(self.num_lots):
                        requests.append(
                            self.modify_order(
                                self.order_ids[asset_name+str(i)][1],
                                asset_name,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.ASK,
                                self.sell_option_q,
                                ask_price+self.lot_expand*(i+1),
                            )
                        ) 
                        orders.append([1,asset_name+str(i)])


            # price put
            flag = "P"
            asset_name = f"UC{strike}{flag}"
            put_bid_price = round(strike-self.underlying_price+bid_price,1)
            put_ask_price = round(strike-self.underlying_price+ask_price,1) 
            res_buy = self.check_valid_trade(1,flag,self.underlying_price, strike, self.time_to_expiry, put_bid_price,self.buy_option_q)
            res_sell = self.check_valid_trade(-1,flag,self.underlying_price, strike, self.time_to_expiry, put_ask_price,self.sell_option_q)
            limits_results_buy = self.check_cum_limits(res_buy[1],res_buy[2],res_buy[3],res_buy[4])
            limits_results_sell = self.check_cum_limits(res_sell[1],res_sell[2],res_sell[3],res_sell[4])
            # If can't buy then sell
            if limits_results_buy["g"][0] or limits_results_buy["v"][0] or limits_results_buy["t"][0]:
                # sell
                if self.positions[asset_name] > -self.position_limit:
                    requests.append(
                        self.modify_order(
                            self.order_ids[asset_name][1],
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.ASK,
                            self.sell_option_q,
                            put_ask_price,
                        )
                    )
                    orders.append([1,asset_name])
                    for i in range(self.num_lots):
                        requests.append(
                            self.modify_order(
                                self.order_ids[asset_name+str(i)][1],
                                asset_name,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.ASK,
                                self.sell_option_q,
                                ask_price+self.lot_expand*(i+1),
                            )
                        ) 
                        orders.append([1,asset_name+str(i)])
                
            # If can't sell then buy
            elif limits_results_sell["g"][0] or limits_results_sell["v"][0] or limits_results_sell["t"][0]:
                # buy   
                if self.positions[asset_name] < self.position_limit:
                    requests.append(
                        self.modify_order(
                            self.order_ids[asset_name][0],
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.BID,
                            self.buy_option_q,  # How should this quantity be chosen?
                            put_bid_price,  # How should this price be chosen?
                        )
                    )
                    orders.append([0,asset_name])
                    for i in range(self.num_lots):
                        requests.append(
                            self.modify_order(
                                self.order_ids[asset_name+str(i)][0],
                                asset_name,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.BID,
                                self.buy_option_q,
                                bid_price-self.lot_expand*(i+1),
                            )
                        ) 
                        orders.append([0,asset_name+str(i)])
            # otherwise do both because it doesn't matter
            else:
                if self.positions[asset_name] < self.position_limit:
                    requests.append(
                        self.modify_order(
                            self.order_ids[asset_name][0],
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.BID,
                            self.buy_option_q,  # How should this quantity be chosen?
                            put_bid_price,  # How should this price be chosen?
                        )
                    )
                    orders.append([0,asset_name])
                    for i in range(self.num_lots):
                        requests.append(
                            self.modify_order(
                                self.order_ids[asset_name+str(i)][0],
                                asset_name,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.BID,
                                self.buy_option_q,
                                bid_price-self.lot_expand*(i+1),
                            )
                        ) 
                        orders.append([0,asset_name+str(i)])
                if self.positions[asset_name] > -self.position_limit:
                    requests.append(
                        self.modify_order(
                            self.order_ids[asset_name][1],
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.ASK,
                            self.sell_option_q,
                            put_ask_price,
                        )
                    )
                    orders.append([1,asset_name])
                    for i in range(self.num_lots):
                        requests.append(
                            self.modify_order(
                                self.order_ids[asset_name+str(i)][1],
                                asset_name,
                                pb.OrderSpecType.LIMIT,
                                pb.OrderSpecSide.ASK,
                                self.sell_option_q,
                                ask_price+self.lot_expand*(i+1),
                            )
                        ) 
                        orders.append([1,asset_name+str(i)])

        # optimization trick -- use asyncio.gather to send a group of requests at the same time
        # instead of sending them one-by-one
        responses = await asyncio.gather(*requests)
        for i in range(0,len(responses)):
            self.order_ids[orders[i][1]][orders[i][0]] = responses[i].order_id

    async def handle_exchange_update(self, update: pb.FeedMessage):

        kind, _ = betterproto.which_one_of(update, "msg")

        if kind == "pnl_msg":
            # When you hear from the exchange about your PnL, print it out
            print("My PnL:", update.pnl_msg.m2m_pnl)

        elif kind == "fill_msg":
            # When you hear about a fill you had, update your positions
            fill_msg = update.fill_msg
            print("Fill: ", fill_msg)
            if fill_msg.order_side == pb.FillMessageSide.BUY:
                self.positions[fill_msg.asset] += update.fill_msg.filled_qty
                
                if len(fill_msg.asset)>2:
                    #self.fills.append([1,fill_msg.asset[-1],int(fill_msg.asset[2:-1]), float(fill_msg.price)])
                    res = self.check_valid_trade(1,fill_msg.asset[-1],self.underlying_price, int(fill_msg.asset[2:-1]), self.time_to_expiry, float(fill_msg.price),fill_msg.filled_qty)
                    self.risks["d"] += res[1]
                    self.risks["g"] += res[2]
                    self.risks["t"] += res[3]
                    self.risks["v"] += res[4]
                    if fill_msg.asset[-1]=="C":
                        # hedge delta by selling delta shares of stock
                        resp = self.place_order(
                            "UC",
                            pb.OrderSpecType.MARKET,
                            pb.OrderSpecSide.ASK,
                            self.risks["d"]
                        )
                    else:
                        # hedge delta by buying delta shares of stock
                        resp = self.place_order(
                            "UC",
                            pb.OrderSpecType.MARKET,
                            pb.OrderSpecSide.BID,
                            abs(self.risks["d"])
                        )
                else:
                    self.risks["d"] += update.fill_msg.filled_qty
            else:
                self.positions[fill_msg.asset] -= update.fill_msg.filled_qty
                if len(fill_msg.asset)>2:
                    #self.fills.append([-1,fill_msg.asset[-1],int(fill_msg.asset[2:-1]), float(fill_msg.price)])
                    res = self.check_valid_trade(-1,fill_msg.asset[-1],self.underlying_price, int(fill_msg.asset[2:-1]), self.time_to_expiry, float(fill_msg.price),fill_msg.filled_qty)
                    self.risks["d"] += res[1]
                    self.risks["g"] += res[2]
                    self.risks["t"] += res[3]
                    self.risks["v"] += res[4]
                    if fill_msg.asset[-1]=="C":
                        # hedge delta by selling delta shares of stock
                        resp = self.place_order(
                            "UC",
                            pb.OrderSpecType.MARKET,
                            pb.OrderSpecSide.BID,
                            abs(self.risks["d"])
                        )
                    else:
                        # hedge delta by buying delta shares of stock
                        resp = self.place_order(
                            "UC",
                            pb.OrderSpecType.MARKET,
                            pb.OrderSpecSide.ASK,
                            self.risks["d"]
                        )
                else:
                    self.risks["d"] -= update.fill_msg.filled_qty
            

        elif kind == "market_snapshot_msg":
            self.update_count += 1
            # When we receive a snapshot of what's going on in the market, update our information
            # about the underlying price.
            book = update.market_snapshot_msg.books["UC"]
            #print(book)
            # Compute the mid price of the market and store it
            self.previous_price = self.underlying_price
            self.underlying_price = (
                float(book.bids[0].px) + float(book.asks[0].px)
            ) / 2
            if self.update_count == self.skip_updates:
                await self.update_options_quotes()
                self.update_count = 0

        elif (
            kind == "generic_msg"
            and update.generic_msg.event_type == pb.GenericMessageType.MESSAGE
        ):
            # The platform will regularly send out what day it currently is (starting from day 0 at
            # the start of the case) 
            self.current_day = float(update.generic_msg.message)
            self.time_to_expiry = (26-self.current_day) / 252
        # If a request to place an order fails, a message of this kind will be sent
        elif kind == "request_failed_msg":
            print("request failed :", update.request_failed_msg)
        elif kind == "trade_msg":
            pass

if __name__ == "__main__":
    start_bot(Case2ExampleBot)