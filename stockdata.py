from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import fix_yahoo_finance as yf
import numpy as np

#Thanks yahoo...
yf.pdr_override() 

#Testing 123...
#Grab the tickers
stocks = ['AAPL', 'MSFT', 'EEM', 'SPY']
wgts = pd.Series([0.25, 0.25, 0.3, 0.2],index=stocks)
pctile = 99

#Number of interpolated points
its = 100

#Past 5 years
end = datetime.datetime.now()-datetime.timedelta(days=1)
start = end - datetime.timedelta(days=5*365)

#Download the goods
df = pdr.get_data_yahoo(stocks, start=start, end=end, adjust_price=True)
df = df.to_frame().unstack(level=1)
#Adjusted Price dataframe
adjprice = df.iloc[:,len(stocks)*4:len(stocks)*5]               
del df

#Rename columns
adjprice.columns = stocks

#Returns
ret = (adjprice / adjprice.shift(1) - 1)*100
ret = ret.iloc[1:]
#Portfolio weighted return
port_ret = ret.dot(wgts)
port_ret = port_ret.iloc[1:]
#Expected return from current weights
port_mean = port_ret.mean()
 
#Individual security and Portfolio x% VaRs
port_hvar = port_ret.quantile(1 - pctile/100)
port_vcvvar = -2.33*port_ret.std()

for column in ret:
    print(ret[column].quantile(1 - pctile/100))

#VCV Matrix and Expected Returns (expected = mean)
vcv = ret.cov().as_matrix()
expret = ret.mean().as_matrix()

sqroot = np.sqrt(np.diag(vcv))

ones = np.ones(len(stocks))

#ABCDs
A = np.dot(np.dot(np.transpose(expret),np.linalg.inv(vcv)),expret)
B = np.dot(np.dot(np.transpose(expret),np.linalg.inv(vcv)),ones)
C = np.dot(np.dot(np.transpose(ones),np.linalg.inv(vcv)),ones)
D = A*C - np.square(B)
#GMV Weights
gmv = np.dot(np.linalg.inv(vcv),ones)/C

#GMV Return
gmv_ret = B/C
#GMV Variance
#gmv_var = 

#Optimal Weights
optimal = np.dot(np.linalg.inv(vcv),expret)/B
#Optimal Return and SD
optimalret = np.dot(np.transpose(optimal),expret)
optimalsd = np.sqrt(np.dot(np.dot(np.transpose(optimal),vcv),optimal))

port_ret_opt = ret.dot(optimal)
port_ret_opt = port_ret_opt.iloc[1:]

#Individual security and Portfolio x% VaRs
port_hvar_opt = port_ret_opt.quantile(1 - pctile/100)
port_vcvvar_opt = -2.33*optimalsd

#frontier returns
minret = np.minimum(expret, optimalret)
maxret = np.maximum(expret, optimalret)
maxsd = np.maximum(sqroot, optimalsd)
i = 0.05
while (i > 0):
    scalingfactor = 1.2 + i
    graphret = np.linspace(min(minret)*0.9,max(maxret)*scalingfactor,its)
    #frontier sd
    graphsd = np.sqrt(((C*np.square(graphret))-2*B*graphret+A)/D)
    if (np.amax(graphsd) >= np.amax(maxsd)):
        i = 0
    else:
        i = i + 0.05

plt.plot(graphsd, graphret)
plt.scatter(sqroot, expret)
plt.plot(graphsd, graphret)
plt.scatter(optimalsd, optimalret, s=100, marker = "*")
plt.scatter(optimalsd, optimalret, s=100, marker = "*", color = "r")
plt.scatter(port_ret.std(), port_mean, s=50, marker = "D", color = "r")

plt.show()
optimal = pd.DataFrame(optimal, index=stocks, columns=['Weights'])
print(optimal)