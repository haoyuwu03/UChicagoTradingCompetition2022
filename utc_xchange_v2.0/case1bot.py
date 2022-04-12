#!/usr/bin/env python

from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import numpy as np

import matplotlib.pyplot as plt
import asyncio

CONTRACTS = ["LBSJ", "LBSM", "LBSQ", "LBSV", "LBSZ"]
BEST_BID_THRESH = (-80, 80)


class TradeBot(UTCBot):
    '''
    '''

    '''
    Things to add
    - edge changing situation
    - adjusting prev based on each contract
    '''

    async def handle_round_started(self):
        self.pnls = []
        # things to adjust for each model - based on year
        self.fairs = {"LBSJ": 213.2, "LBSM": 207.55, "LBSQ": 209.6,
                      "LBSV": 280.91, "LBSZ": 281.31}
        self.prev = self.fairs.copy()
        self.year = 2
        self.rain = 19.87
        self.up_year = 0
        # things to adjust for each model

        self.day_till = {"LBSJ": 83, "LBSM": 125, "LBSQ": 167,
                         "LBSV": 209, "LBSZ": 251}
        self.day_start = self.day_till.copy()

        self.month = 0
        self.day = 0
        self.pos = {}
        self.order_ids = {}

        self.params = {}
        self.bid_size = {}
        self.ask_size = {}
        self.edge = {}

        self.bestbids = {k: v - 1 for k, v in self.fairs.items()}
        self.bestasks = {k: v + 1 for k, v in self.fairs.items()}

        # to keep time
        self.ticking = 1
        self.complete = False

        for contract in CONTRACTS:

            self.pos[contract] = 0
            self.order_ids[contract + "_bid"] = ""
            self.order_ids[contract + "_ask"] = ""
            self.order_ids[contract + "_bidlot1"] = ""
            self.order_ids[contract + "_asklot1"] = ""
            self.order_ids[contract + "_bidlot2"] = ""
            self.order_ids[contract + "_asklot2"] = ""

            if contract not in self.params:
                self.params[contract] = {}

            self.params[contract]["edge"] = 0.5
            self.params[contract]["fade"] = 5
            self.params[contract]["slack"] = 0
            self.params[contract]["size"] = 15
            self.params[contract]["lot_dif"] = 5

            self.bid_size[contract] = self.params[contract]["size"]
            self.ask_size[contract] = self.params[contract]["size"]
            self.edge[contract] = self.params[contract]["edge"]
        self.pos_hist = [self.pos.copy()]
        self.fairs_hist = [self.fairs.copy()]
        self.fade_totals = {contract: 0 for contract in CONTRACTS}
        self.fairs_faded = self.fairs.copy()
        asyncio.create_task(self.update_quotes())

    def april_model(self, month, next_rain, year, up_year, prev, april_dist):
        return 35.0195 + 0.0347 * next_rain + 0.1021 * month + 1.3181 * year + 0.7873 * up_year + 0.8322 * prev + 0.0058 * april_dist

    def june_model(self, month, next_rain, year, up_year, prev, june_dist):
        return 88.6061 - 0.2538 * next_rain + 0.3405 * month + 5.0820 * year - 1.2535 * up_year - 0.5713 * prev + 0.0134 * june_dist

    def aug_model(self, month, next_rain, year, up_year, prev, aug_dist):
        return 165.5746 - 0.2772 * next_rain + 1.3795 * month + 13.7199 * year + 4.4117 * up_year + 0.1295 * prev + 0.0617 * aug_dist

    def oct_model(self, month, next_rain, year, up_year, prev, oct_dist):
        return 126.5637 - 0.1292 * next_rain + 0.7739 * month + 5.5137 * year + 8.3913 * up_year - 0.4477 * prev + 0.0362 * oct_dist

    def dec_model(self, month, next_rain, year, up_year, prev, dec_dist):
        return 222.1475 - 0.0355 * next_rain + 0.0039 * month + 13.7557 * year + 33.9767 * up_year - 0.0145 * prev + 0.0003 * dec_dist

    def rain_model(self, rain, month_squared, year):
        return 150.2379 + 2.447 * rain + 1.6158 * month_squared + 13.234 * year

    def calculate_fairs(self):
        '''
        Recreate model here
        '''
        model_dict = {"LBSJ": self.april_model, "LBSM": self.june_model,
                      "LBSQ": self.aug_model, "LBSV": self.oct_model,
                      "LBSZ": self.dec_model}

        for contract in CONTRACTS:

            self.ask_size[contract] = round(self.params[contract]["size"])
            self.bid_size[contract] = round(self.params[contract]["size"])

            w1 = self.day_till[contract] / self.day_start[contract]
            w2 = (1 / (self.day_till[contract] + 0.001) ** 0.1)
            w3 = (1 / (self.day_till[contract] + 0.001) ** 0.1) / 3

            length = w1 + w2 + w3

            w1 = w1 / length
            w2 = w2 / length
            w3 = w3 / length

            model1 = model_dict[contract](self.month, self.rain,
                                          self.year, self.up_year,
                                          self.prev[contract], self.day_till[contract])

            model2 = self.prev[contract]

            edited_month = (self.month - 5.5) ** 2

            model3 = self.rain_model(self.rain, edited_month, self.year)

            self.fairs[contract] = w1 * model1 + w2 * model2 + w3 * model3
            self.fairs_faded = {k: v + self.fade_totals[k] for k, v in self.fairs.items()}
        self.complete = False

    def update_fade(self):
        '''
        Adjust fairs based on positions
        For every 10 increase in lots, increase/decrease fair by fade parameter
        '''
        for contract in CONTRACTS:
            # if BEST_BID_THRESH[0] < self.pos[contract] < BEST_BID_THRESH[1]:
            if self.pos[contract] > 80:
                self.fade_totals[contract] -= self.params[contract]["fade"]
            elif self.pos[contract] < -80:
                self.fade_totals[contract] += self.params[contract]["fade"]
            else:
                if self.fade_totals[contract] > 0:
                    self.fade_totals[contract] -= self.params[contract]["fade"]
                elif self.fade_totals[contract] < 0:
                    self.fade_totals[contract] += self.params[contract]["fade"]

    def update_fairs(self):
        '''
        You should implement this function to update the fair value of each asset as the
        round progresses.
        '''
        if (self.complete):
            self.calculate_fairs()
        if (self.ticking == 0):
            self.update_fade()

    async def update_quotes(self):
        '''
        This function updates the quotes at each time step. In this sample implementation we
        are always quoting symetrically about our predicted fair prices, without consideration
        for our current positions. We don't reccomend that you do this for the actual competition.
        '''

        while True:

            self.update_fairs()

            for contract in CONTRACTS:
                print(contract, self.pos[contract], self.fairs[contract])
                '''
                This sets up a ratio to make sure that positions are not too one-sided. 
                Essentially, as abs(position) goes past 50, there will be diminishing 
                number of bids/ask quantities depending on amount
                '''

                if self.day_till[contract] < 2:
                    self.cancel_order(contract + '_bid')
                    self.cancel_order(contract + '_ask')
                    self.cancel_order(contract + '_asklot1')
                    self.cancel_order(contract + '_asklot2')
                    self.cancel_order(contract + '_bidlot1')
                    self.cancel_order(contract + '_bidlot2')

                    CONTRACTS.remove(contract)
                    continue

                bid_response = await self.modify_order(
                    self.order_ids[contract + '_bid'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.BID,
                    self.bid_size[contract],
                    max(0, round(self.fairs_faded[contract] - self.edge[contract], 2)))
                bid_lot1 = await self.modify_order(
                    self.order_ids[contract + '_bidlot1'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.BID,
                    self.bid_size[contract],
                    max(0, round(self.fairs_faded[contract] - self.params[contract]["lot_dif"] - self.edge[contract], 2)))
                bid_lot2 = await self.modify_order(
                    self.order_ids[contract + '_bidlot2'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.BID,
                    self.ask_size[contract],
                    max(0, round(self.fairs_faded[contract] - 2 * self.params[contract]["lot_dif"] - self.edge[contract], 2)))
                ask_response = await self.modify_order(
                    self.order_ids[contract + '_ask'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.ASK,
                    self.ask_size[contract],
                    max(0, round(self.fairs_faded[contract] + self.edge[contract], 2)))
                ask_lot1 = await self.modify_order(
                    self.order_ids[contract + '_asklot1'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.ASK,
                    self.ask_size[contract],
                    max(0, round(self.fairs_faded[contract] + self.params[contract]["lot_dif"] + self.edge[contract], 2)))
                ask_lot2 = await self.modify_order(
                    self.order_ids[contract + '_asklot2'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.ASK,
                    self.bid_size[contract],
                    max(0, round(self.fairs_faded[contract] + 2 * self.params[contract]["lot_dif"] + self.edge[contract], 2)))

                assert bid_response.ok
                self.order_ids[contract + '_bid'] = bid_response.order_id

                assert ask_response.ok
                self.order_ids[contract + '_ask'] = ask_response.order_id

                assert bid_lot1.ok
                self.order_ids[contract + '_bidlot1'] = bid_lot1.order_id

                assert bid_lot2.ok
                self.order_ids[contract + '_bidlot2'] = bid_lot2.order_id

                assert ask_lot1.ok
                self.order_ids[contract + '_asklot1'] = ask_lot1.order_id

                assert ask_lot2.ok
                self.order_ids[contract + '_asklot2'] = ask_lot2.order_id

            await asyncio.sleep(0.1)

            self.ticking += 1
            if (self.ticking > 5):
                self.ticking = 0

    async def handle_exchange_update(self, update: pb.FeedMessage):
        '''
        This function receives messages from the exchange. You are encouraged to read through
        the documentation for the exachange to understand what types of messages you may receive
        from the exchange and how they may be useful to you.

        Note that monthly rainfall predictions are sent through Generic Message.
        '''
        kind, _ = betterproto.which_one_of(update, "msg")
        if kind == "pnl_msg":
            print('Realized pnl:', update.pnl_msg.realized_pnl)
            print("M2M pnl:", update.pnl_msg.m2m_pnl)
            self.pnls.append(float(update.pnl_msg.realized_pnl))
            self.pos_hist.append(self.pos.copy())
            self.fairs_hist.append(self.fairs.copy())

            # if self.day != 0 and self.day % 21 == 0:
            if self.day == 250:
                fig, ax = plt.subplots(2, 5, figsize=(20, 8))
                j = 0
                for contract in ["LBSJ", "LBSM", "LBSQ", "LBSV", "LBSZ"]:
                    ax[0, j].plot([i[contract] for i in self.fairs_hist])
                    ax[1, j].plot([i[contract] for i in self.pos_hist])
                    j += 1
                plt.show()
            for contract in CONTRACTS:
                self.day_till[contract] -= 1
            self.complete = True
            print("Day:", self.day)
            self.day += 1

        elif kind == "market_snapshot_msg":
            # Updates your record of the Best Bids and Best Asks in the market
            # print(update.market_snapshot_msg)
            if (not self.complete):
                for contract in CONTRACTS:
                    book = update.market_snapshot_msg.books[contract]
                    bids = len(book.bids)
                    asks = len(book.asks)
                    if bids != 0 and asks != 0:
                        our_bid = self.fairs[contract] - self.edge[contract]
                        our_ask = self.fairs[contract] + self.edge[contract]

                        if abs(float(book.bids[0].px) - our_bid) < 0.01:
                            self.bestbids[contract] = our_bid - 0.02
                        else:
                            # idx = int(len(book.bids) * 0.2)
                            # self.bestbids[contract] = min(float(book.bids[idx].px), self.fairs[contract] - self.edge[contract] + 50)
                            self.bestbids[contract] = float(book.bids[0].px)

                        if abs(float(book.bids[0].px) - our_ask) < 0.01:
                            self.bestasks[contract] = our_ask + 0.02
                        else:

                            # idx = int(len(book.asks) * 0.2)
                            # self.bestasks[contract] = max(float(book.asks[idx].px), self.fairs[contract] + self.edge[contract] - 50)
                            self.bestasks[contract] = float(book.asks[0].px)

                        mid_bid = float((book.bids[int(len(book.bids) / 4)]).px)
                        mid_ask = float((book.asks[int(len(book.asks) / 4)]).px)
                        self.prev[contract] = (mid_bid + mid_ask) / 2

                    if bids < 3 and asks < 3:
                        self.edge[contract] = self.params[contract]["edge"] + self.params[contract]["slack"]
                    elif bids > 15 and asks > 15:
                        self.edge[contract] = self.params[contract]["edge"] - self.params[contract]["slack"]
                    else:
                        self.edge[contract] = self.params[contract]["edge"]

        elif kind == "fill_msg":
            # When you hear about a fill you had, update your positions
            fill_msg = update.fill_msg

            if fill_msg.order_side == pb.FillMessageSide.BUY:
                self.pos[fill_msg.asset] += update.fill_msg.filled_qty
            else:
                self.pos[fill_msg.asset] -= update.fill_msg.filled_qty

        elif kind == "generic_msg":
            # Saves the predicted rainfall
            try:
                pred = float(update.generic_msg.message)
                self.rain = pred
                self.month += 1
            # Prints the Risk Limit message
            except ValueError:
                pass
                # print(update.generic_msg.message)
        '''
        elif kind == "trade_msg":
            update = update.trade_msg
            print("Asset " + update.asset + " was traded at price " + update.price + " quantity " + str(update.qty))
            print("This occured at time stamp " + update.timestamp)
        '''


if __name__ == "__main__":
    start_bot(TradeBot)