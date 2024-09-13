import pickle
import asyncio
import datetime
import winsound
import time
import yfinance as yf
import numpy as np
from tastytrade import DXLinkStreamer, ProductionSession, Account
from tastytrade.dxfeed import EventType
from tastytrade.order import NewOrder, OrderAction, Decimal, OrderType, OrderTimeInForce, PriceEffect, NewComplexOrder
from tastytrade import DXLinkStreamer
from tastytrade.instruments import get_option_chain, OptionType
from os import system

def createSession():
    ###  ACCOUNT ACCESS INFO - REMOVED FOR PUBLIC REPO ###
    session = ProductionSession('account_name',"access_key",) # would need to replace name and key with actual for your account

    # Save the session object to a file
    with open('session.pkl', 'wb') as file:
        pickle.dump(session, file)

    print(f"Session created for {datetime.datetime.now().date()}",' '*30)
    return session

def TradeCheck():
    ### Criteria Logic used by previou build - Removed in current build ###
    print('Checking trade criteria...', end='\r')
    # Define the ticker symbol for S&P 500
    ticker = "^GSPC"

    # Download the last 2 years of historical data
    data = yf.download(ticker, period="3mo", interval="1d",progress=False)

    # Drop NaN values that result from shifting
    data.dropna(inplace=True)

    move_c = data['Close'].to_numpy()[-12:-1]
    change = []
    for x in range(move_c.shape[0]-1):
        change.append(abs(move_c[x]/move_c[x+1]-1)*100)
    moving_avg = np.average(change[:10])
    if moving_avg < 0.6:
        print('Trade Check Passed',' '*30)
    else:
        print('Trade Criteria Not Met... Exiting',' '*30)
        time.sleep(10)
    return moving_avg < 0.6

def loadSession():
    # Loads the latest session ID for TastyTrade API calls
    with open('session.pkl', 'rb') as file:
        session = pickle.load(file)
    return session

async def get_price(session,symbol):
    # Contacts TatsyTrade for current pricing
    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(EventType.QUOTE, symbol)
        quote = await streamer.get_event(EventType.QUOTE)
        return ((quote.askPrice + quote.bidPrice)/2)

def get_butterfly_price(session,sp,sc,lp,lc):
    # Gets the pricing for the Short Calls/Puts and Long Calls/Puts as defined in the parameters
    price = asyncio.run(get_price(session,[sp.streamer_symbol])) + asyncio.run(get_price(session,[sc.streamer_symbol])) - asyncio.run(get_price(session,[lp.streamer_symbol])) - asyncio.run(get_price(session,[lc.streamer_symbol]))
    price = round(price*20)/20 - (0.05)
    return price

async def day_open(session,symbol):
    # Gets the opening SPX price for the day - no longer used
    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(EventType.SUMMARY, symbol)
        quote = await streamer.get_event(EventType.SUMMARY)
        return (quote.dayOpenPrice)
  
async def get_last_price(session,symbol):
    # Gets the last printed price for the specificed symbol
    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(EventType.TRADE, symbol)
        quote = await streamer.get_event(EventType.TRADE)
        return (quote.price)

def BTTRFLY_SPX(just_price=False):
    ### Main Order Logic ###
    session = loadSession()

    SPX_Price = asyncio.run(get_last_price(session,['SPX']))

    if just_price:
            return SPX_Price

    short_price = round((SPX_Price)/5)*5

    ### Select EOD Options ###
    chain = get_option_chain(session, 'SPX')
    next_options = chain[(datetime.datetime.today()).date()]

    for o in next_options:
        if o.strike_price == short_price and o.option_type == OptionType.PUT:
            sp = o
        if o.strike_price == short_price and o.option_type == OptionType.CALL:
            sc = o
        if o.strike_price == short_price - Decimal(35) and o.option_type == OptionType.PUT:
            lp = o
        if o.strike_price == short_price + Decimal(35) and o.option_type == OptionType.CALL:
            lc = o

    print(f'Butterfly POS: {lp.strike_price}-{sp.strike_price}-{lc.strike_price}')
    
    ### Get Selected Options Pricing ###
    price = get_butterfly_price(session,sp,sc,lp,lc)

    ### Contruct Order ###
    order=NewOrder(
        time_in_force=OrderTimeInForce.DAY,
        order_type=OrderType.LIMIT,
        legs=[
            lp.build_leg(Decimal(1),OrderAction.BUY_TO_OPEN),
            sp.build_leg(Decimal(1),OrderAction.SELL_TO_OPEN),
            sc.build_leg(Decimal(1),OrderAction.SELL_TO_OPEN),
            lc.build_leg(Decimal(1),OrderAction.BUY_TO_OPEN)
            ],
        price=Decimal(price),
        price_effect=PriceEffect.CREDIT
    )

    ### Notify user that an order was placed
    winsound.PlaySound('notification.mp3',winsound.SND_FILENAME)
    return price,sp,sc,lp,lc,short_price

def have_open_orders():
    # Checks for any open orders - no used in current build
    session = loadSession()
    account = Account.get_account(session,'5WT58311')
    orders = account.get_live_orders(session)
    spx_orders = []
    for o in orders:
        if o.underlying_symbol == 'SPX':
            spx_orders.append(o)
    if len(spx_orders) > 0:
        return True
    else:
        return False

if __name__ == '__main__':
    system("title SPX Open Butterfly Trade Tracker")
    running = True
    session = createSession()
    trade_on = False
    try:
        profit = np.load('EOD_profit.npy')
        balance = np.load('EOD_balance.npy')
    except:
        profit = []
        balance = Decimal(0)

    print('SPX Open Butterfly Trade Tracker Initialized')
    while running:
        try:
            now = datetime.datetime.now()
            hour = now.hour
            minute = now.minute
            second = now.second
            day = now.weekday()

            ### Restart Session ###
            if hour == 0 and minute == 0 and second == 1 and day < 5:
                time.sleep(1)
                session = createSession()
                time.sleep(10)

            ### Trade Logic ###
            if hour == 7 and minute == 31 and second == 0 and day < 5:
                print('  '*20,end='\r')
                price, sp, sc, lp, lc, short_strike = BTTRFLY_SPX()
                print(f'OPENING BUTTERFLY TRADE @ {price} CREDIT')
                trade_on = True
                time.sleep(10)
                while trade_on:
                    curr_price = get_butterfly_price(session,sp,sc,lp,lc)
                    curr_underlying = BTTRFLY_SPX(just_price=True)
                    if price - curr_price >= 2.5:
                        balance += int((price - curr_price)*100)
                        print(f'WINNER {int((price - curr_price)*100)}',' '*20)
                        trade_on = False
                    elif abs(short_strike - curr_underlying) >= 28:
                        balance -= int((price - curr_price)*100)
                        print(f'LOSER {int((price - curr_price)*100)}',' '*20)
                        trade_on = False
                    else:
                        print(f'TRADE ON: {int((price - curr_price)*100)}',' '*10,end='\r')
                    time.sleep(5)
                profit.append(balance)
                np.save('EOD_profit.npy',profit)
                np.save('EOD_balance.npy',balance)
                time.sleep(60)
            else:
                curr_time = datetime.datetime.now()
                time_string = '07:31:00'
                time_format = '%H:%M:%S'
                trade_time = datetime.datetime.strptime(time_string, time_format).time()
                trade_time = datetime.datetime.combine(datetime.datetime.now().date(), trade_time)
                time_delta = trade_time - curr_time
                if time_delta < datetime.timedelta(0):
                    if day < 4:
                        time_delta = time_delta + datetime.timedelta(days=1)
                    else:
                        time_delta = time_delta + datetime.timedelta(days=7-day)
                rounded_seconds = round(time_delta.total_seconds())
                time_delta = datetime.timedelta(seconds=rounded_seconds)

                print(f'{datetime.datetime.now().date()} Time until trade: {time_delta}',' '*10,end='\r')
                time.sleep(1)
        except Exception as error:
            print(f'Failed @ {now}: {error}')

