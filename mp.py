from mpi4py import MPI
import time
import akshare as ak
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dateutil
import datetime
import random
import os


# 计算杨辉三角
def triangle(n):
    N = [1]
    for i in range(n):  #打印n行
        N.append(0)
        N = [N[k] + N[k-1] for k in range(i+2)]
    N = np.array(N)
    return N

def EXp_N(L,n):
    l = np.linspace(0,L,n)
    N = np.exp(-l)
    return N

if True:
    matplotlib.rcParams['axes.linewidth']       = 1
    matplotlib.rcParams['xtick.major.size']     = 6
    matplotlib.rcParams['xtick.major.width']    = 1
    matplotlib.rcParams['xtick.minor.size']     = 6
    matplotlib.rcParams['xtick.minor.width']    = 1
    matplotlib.rcParams['ytick.major.size']     = 6
    matplotlib.rcParams['ytick.major.width']    = 1
    matplotlib.rcParams['ytick.minor.size']     = 6
    matplotlib.rcParams['ytick.minor.width']    = 1
    matplotlib.rcParams['ytick.direction']     = 'in'
    matplotlib.rcParams['xtick.direction']    = 'in'

    matplotlib.rcParams['xtick.major.pad']      = 10
    matplotlib.rcParams['ytick.major.pad']      = 10

    matplotlib.rcParams['mathtext.default']  = 'regular'
#-- End

# API for aks
# 个股信息查询
def search_code(code,print_or_not):
    try:
        stock = ak.stock_individual_info_em(symbol=code)
    except:
        print('Not found')
    if print_or_not == 1:
        print(stock)
    return stock
# 所有A股上市公司的实时行情数据
def all_in_A():
    all = ak.stock_zh_a_spot_em()
    return all

def trade_get():
    enable_hist_df = ak.tool_trade_date_hist_sina()
    enable_hist_df.columns = ['list','trade_date',]
    enable_hist_df['trade_date'] = pd.to_datetime(enable_hist_df['trade_date'])
    return enable_hist_df

def get_stock(code,fq):
    # 获取并保存数据
    try:
        f = open('./storehouse/'+code+'.csv','r')
        stock_df = pd.read_csv(f)
    except:
        print(1)
        stock_df = ak.stock_zh_a_hist(symbol=code, adjust=fq).iloc[:, :6]
        stock_df.columns = [
            'date',
            'open',
            'close',
            'high',
            'low',
            'volume',
        ]
        stock_df.to_csv('./storehouse/'+code + '.csv')
    stock_df.index = pd.to_datetime(stock_df['date'])
    stock_df.index_col = 'date'
    return stock_df

# 获取一定范围内的数据
# n-i，n
def Stock_range(stock_df,td,count):
    list = stock_df['date'].tolist()
    try:
        index = list.index(td.strftime('%Y-%m-%d'))
        stock_range = stock_df[stock_df['date'].between(stock_df['date'][index-count+1],stock_df['date'][index-1+1])]
    except:
        stock_range = []
    # print(td,stock_range)
    return stock_range
# n-i，n-1
def Stock_range2(stock_df,td,count):
    list = stock_df['date'].tolist()
    try:
        index = list.index(td.strftime('%Y-%m-%d'))
        stock_range = stock_df[stock_df['date'].between(stock_df['date'][index-count],stock_df['date'][index-1])]
    except:
        stock_range = []
    # print(td,stock_range)
    return stock_range

# 预留，上面函数似乎过于复杂，留待简化
def Stock_hist(td,code,count):
    return

# 后续考虑滑点
def get_today_data_old(Context,code):
    today = Context.dt.strftime('%Y-%m-%d')
    da = get_stock(code,Context.fq)
    data = da[da['date']==today][:]
    return data

def get_today_data(Context,code):
    today = Context.dt.strftime('%Y-%m-%d')
    data = g.da[g.da['date']==today][:]
    return data

# 已经考虑停牌，后续考虑分红
def order_root(Context,today_price,code,amount,o_or_c):

    if len(today_price) == 0:
        print(f"\033[33m{'今日停牌！'}\033[0m")
        return
    ymd = today_price['date'][0]
    # 应在底层下单函数中考虑滑点
    logi = random.choice([-1,1])
    today_price = today_price[o_or_c][0]*(1+logi*0.5/100)
    if amount>0:
        if Context.cash - amount*today_price < 0:
            amount = int(int(Context.cash/today_price)/100)
            if amount == 0:
                print(f"\033[31m{ymd}\033[0m",f"\033[33m{':现金严重不足，无法买入！！！'}\033[0m")
            else:
                amount = amount*100
                print(f"\033[31m{ymd}\033[0m",f"\033[33m{':现金不足，已帮您调整为%d' % (amount)}\033[0m")
        else:
            amount = int(amount/100)
            amount = amount*100
            print(f"\033[31m{ymd}\033[0m",f"\033[33m{':现金充足，已做整数调整，调整后买入%d' % (amount)}\033[0m")

    else:
        if amount+Context.positions.get(code,0)<=0:
            if amount+Context.positions.get(code,0)<0:
                amount = -Context.positions.get(code,0)
                print(f"\033[31m{ymd}\033[0m",f"\033[33m{':持仓不足，全仓卖出！'}\033[0m")
            elif Context.positions.get(code,0)==0:
                amount = -Context.positions.get(code,0)
                print(f"\033[31m{ymd}\033[0m",f"\033[33m{':持仓为0，无法卖出！'}\033[0m")
            elif amount+Context.positions.get(code,0)==0:
                amount = -Context.positions.get(code,0)
                print(f"\033[31m{ymd}\033[0m",f"\033[33m{':持仓刚好，全仓卖出！'}\033[0m")

        else:
            amount = int(amount/100)
            amount = amount*100
            print(f"\033[31m{ymd}\033[0m",f"\033[33m{':持仓充足，已做整数调整，调整后卖出%d' % -amount}\033[0m")


    if amount != 0:
        if amount>0:
            # 买入手续费，按全佣上限0.2%
            service = abs(amount)*today_price*0.2/100
        else:
            # 卖出手续费，按全佣上限0.2%
            service = abs(amount)*today_price*0.2/100
        if service <= 5:
            service = 5
        Context.cash -= service
    else:
        service = 0

    Context.cash -= amount*today_price

    print('          ',f"\033[30m{'Service charge:'}\033[0m",round(service,2))
    Context.positions[code] = Context.positions.get(code,0)+amount
    # print(Context.positions)
    if Context.positions[code] == 0:
        del Context.positions[code]
    return

def order(Context,code,amount,o_or_c,today_price = 0):
    if today_price == 0:
        today_price = get_today_data(Context,code)
    order_root(Context,today_price,code,amount,o_or_c)

def order_target(Context,code,amount,o_or_c,today_price = 0):
    if amount<0:
        print('数量不能为负，已调整为0')
    if today_price == 0:
        today_price = get_today_data(Context,code)
    hold_amount = Context.positions.get(code,0)
    delta_amount = amount - hold_amount
    order_root(Context,today_price,code,delta_amount,o_or_c)

def order_value(Context,code,value,o_or_c,today_price = 0):
    if today_price == 0:
        today_price = get_today_data(Context,code)
    amount = int(value/today_price[o_or_c][0])
    order_root(Context,today_price,code,amount,o_or_c)

def order_target_value(Context,code,value,o_or_c,today_price = 0):
    if value<0:
        print('价值不能为负，已调整为0')

    if today_price == 0:
        today_price = get_today_data(Context,code)
    hold_value = Context.positions.get(code,0)*today_price[o_or_c][0]
    delta_value = value - hold_value
    order_value(Context,code,delta_value,o_or_c)

# 非常重要，这是一个全局类，用于方便用户在初始化函数和策略函数里随心所欲地定义变量，这些变量都会被存在g的属性里
class G:
    pass
g = G()
# 交易判断器
g.deal = 0

# 回测函数
def run(Context):
    Init(Context)
    init_ben = benchmark(Context)
    init_cash = Context.cash
    plt_value = pd.DataFrame(index=pd.to_datetime(Context.date_range['trade_date']),columns=['value'])
    last_prize = {}

    for td in Context.date_range['trade_date']:
        # print('Today:',td)
        Signal=2
        Context.dt = dateutil.parser.parse(str(td))


        Cash = Context.cash

        for stock_code in Context.positions:
            today_p = get_today_data(Context,stock_code)
            if len(today_p) == 0:
                p = last_prize[stock_code]
            else:
                p = today_p['close'][0]
                last_prize[stock_code] = p
            Cash += p*Context.positions[stock_code]

        plt_value.loc[td,'value'] = Cash

        # handle(Context,td)
        # 判断今日是否交易
        # print(td,g.deal)
        if g.deal == 0:
            ma,ma20,Signal = handle(Context,td,plt_value)
            if ma != 0:
                plt_value.loc[td,'ma'] = (ma-init_ben)/init_ben
                plt_value.loc[td,'ma20'] = (ma20-init_ben)/init_ben
        else:
            g.deal -= 1
        # print(g.deal)

        # benchmark
        today_p_ben = get_today_data(Context,Context.benchmark)
        if len(today_p_ben) == 0:
            p2 = last_prize[Context.benchmark]
        else:
            p2 = today_p_ben['close'][0]
            last_prize[Context.benchmark] = p2
        plt_value.loc[td,'value_ben'] = p2
        # print(Cash)

        # 记录交易信号
        if Signal==1:
            plt_value.loc[td,'buy'] = (p2-init_ben)/init_ben
        elif Signal==0:
            plt_value.loc[td,'sell'] = (p2-init_ben)/init_ben

    # plot
    plt_value['return'] = (plt_value['value']-init_cash) / init_cash
    plt_value['bench_self'] = (plt_value['value_ben'] - init_ben) / init_ben
    # plt_value[['return','benchmarker']].plot()
    plt_value[['return','bench_self','ma','ma20']].plot()
    # plt.scatter(plt_value.index,plt_value['buy'],color='r',zorder=10)
    # plt.scatter(plt_value.index,plt_value['sell'],color='g',zorder=10)
    #set_benchmark2(Context)
    plot_return(Context)
    plt_value.to_csv('./fb/'+str(g.k1)+'_'+str(g.k2)+'.csv')
    """ if np.sum(plt_value['return'][:])< 0.03:
    #if plt_value['return'][-1]< 0.03:
        return -1
    elif np.sum(plt_value['return'][:])> 0.1:
    #elif plt_value['return'][-1]> 0.2:
        return 1
    else:
        return 0 """
    return plt_value['return'][-1]



def plot_return(Context):
    plt.legend()
    plt.axhline(y=0,c='grey',ls='--',lw=1,zorder=0)
    plt.grid(alpha=0.4)
    plt.xlabel(u'Date',fontsize=16)
    plt.ylabel(u'Return',fontsize=16)
    #plt.semilogy()
    #plt.show()
    plt.savefig('./fb/'+str(g.k1)+'_'+str(g.k2)+'.png',bbox_inches='tight', dpi=64)
    plt.close()


def set_benchmark(Context,code):
    Context.benchmark = code

def set_benchmark2(Context):
    df = ak.stock_a_pe(symbol=Context.benchmark2)
    df.index = pd.to_datetime(df['date'])
    try:
        df = df['close'][Context.date_start:Context.date_end]
        df = (df[:]-df[0])/df[:]
        plt.plot(df.index,df,label=Context.benchmark2)
    except:
        print('无法获取benchmark')
        return

def benchmark(Context):
    stock_df = get_stock(Context.benchmark,Context.fq)
    stock_range = stock_df[stock_df['date'].between(Context.date_start,Context.date_end)]
    if len(stock_range['close'])==0:
        print(f"\033[31m{'无法获取历史数据，回测失败！起始时间为'}\033[0m",stock_df['date'][0])
        os.kill()
    # print(stock_range)
    else:
        init_r = stock_range['close'][0]
        return init_r

class Context:
    def __init__(self, cash, date_start, date_end, fq, bench):
        self.cash = cash
        self.date_start = date_start
        self.date_end = date_end
        self.fq = fq
        self.positions = {}
        self.benchmark = '00'
        self.benchmark2 = bench
        self.date_range = enable_hist_df[enable_hist_df['trade_date'].between(date_start,date_end)]
        self.dt = None  # dateutil.parser.parse(date_start)

def print_end():
    print('初始化完成')









# 用户函数：--------------------------------------------------------------------------------

def Init(Context):
    # g.code = '601390'
    g.init = set_benchmark(Context,g.code)
    g.c = 'close'
    g.m1 = triangle(59)
    g.m2 = triangle(179)
    g.k = 2
    # g.k1 = 10
    # g.k2 = 70
    g.cash = Context.cash
    print_end()
    pass

# ---------------------------------------------
enable_hist_df = pd.read_csv('history.csv',parse_dates=['trade_date'])
enable_hist_df.columns = [
    'list',
    'trade_date',
]
# ---------------------------------------------


def handle(Context,td,value):
    list = Context.date_range['trade_date'].tolist()
    index = list.index(td)
    # 交易信号
    Signal=2
    # 是否要考虑停牌？
    history = Stock_range2(g.da,td,80)
    if len(history) == 0:
        print(f"\033[34m{'今日停牌，不交易'}\033[0m")
        return 0,0,Signal
    else:
        ma5 = history['close'][-g.k2:].mean()
        ma20 = history['close'][-g.k1:].mean()
        # history['close'][-2]-history['close'][-1]>0
        if 0.98*ma5>ma20 and g.code not in Context.positions:
            order_value(Context,g.code,Context.cash,g.c)
            Signal = 1
        elif ma5<ma20 and g.code in Context.positions:
            order_target(Context,g.code,0,g.c)
            Signal = 0
    return ma5,ma20,Signal


def handle2(Context,td,value):
    Signal=2
    # 是否要考虑停牌？
    history = Stock_range2(get_stock(g.code,Context.fq),td,20)
    if len(history) == 0:
        print(f"\033[34m{'今日停牌，不交易'}\033[0m")
        return 0,0,Signal
    else:
        ma = history['close'][-20:].mean()
        #m = history['close'][0]
        up = ma + g.k*history['close'].std()
        low = ma - g.k*history['close'].std()

        p = get_today_data(Context,g.code)['close'][0]
        #print(td,p-low)
        cash = Context.cash
        if p <= low and g.code not in Context.positions:
            print('买入',p)
            order_value(Context,g.code,g.cash,g.c)
            Signal = 1

        elif p >= up and g.code in Context.positions:

            order_target(Context,g.code,Context.positions[g.code]/2,g.c)
            Signal = 0
            print('卖出',p)
    return up,low,Signal


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start = time.time()
g.code = '600133'
g.da = get_stock(g.code,'hfq')
i1,i2 = 10,70
j1,j2 = 10,70
m = i2-i1
n = j2-j1
re = np.zeros((m, n))
mi = m/(size-1)
print(mi)
if rank >= 0 and rank != size-1:
    sumi = np.zeros((m,n))
    for ii in range(int(mi*rank), int(mi*(rank+1))):
        for jj in range(j1-10, j2-10):
            if jj != ii and abs(jj-ii) >= 3:
                g.k1 = ii
            g.k2 = jj
            print(g.k1, g.k2)
            C = Context(100000,'2012-04-01','2023-04-01','hfq',"000300.XSHG")
            sumi[ii-i1, jj-j1] = run(C)
    comm.send(sumi, dest=size-1)

elif rank == size-1:
    # print("------")
    s = np.zeros((m,n))
    for j in range(0,size-1):
        s1 = comm.recv(source=j)
        s += s1
    print('sum:',s)
    re = s
    np.savetxt('./'+str(i1)+'_'+str(i2)+'v'+str(j1)+'_'+str(j2)+'.txt', re,fmt='%f',delimiter=',')
    y = np.linspace(i1, i2-1, i2-i1)
    x = np.linspace(j1, j2-1, j2-j1)
    X, Y = np.meshgrid(x,y)
    plt.contourf(X,Y,re,cmap='RdYlBu_r')
    plt.colorbar()
    plt.savefig('./'+str(i1)+'_'+str(i2)+'v'+str(j1)+'_'+str(j2)+'.png',bbox_inches='tight', dpi=128)
    print("rank %d:" % rank)
end = time.time()

print('Running time: %s Seconds'%(end-start))
