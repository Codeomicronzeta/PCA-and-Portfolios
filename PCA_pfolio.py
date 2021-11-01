# The aim of this project is to use dimesionality reduction technique Principal Component Analysis (PCA) 
#for optimization and diversification of portfolio consisting of Mutual Funds.
# The dataset used consists of Net Asset Value (NAV) of 40 Mutual Funds. The data has been taken from Moneycontrol.com

# 1.1. Loading the python packages

# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#Import Model Packages 
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import inv, eig, svd

from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
import warnings
warnings.filterwarnings('ignore')

# 1.2. Loading the Data
# load dataset
dataset = read_csv('MF_5_cdata.csv',index_col=0)

type(dataset)

# 2. Exploratory Data Analysis
# 2.1. Descriptive Statistics
# shape
dataset.shape
set_option('display.width', 100)
dataset.head(5)

# types
set_option('display.max_rows', 500)
dataset.dtypes

# describe data
set_option('precision', 3)
dataset.describe()

# 3. Data Preparation
# 3.1. Data Cleaning
# Let us check for the NAs in the rows, either drop them or fill them with the mean of the column

#Checking for any null values and removing the null values'''
print('Null Values =',dataset.isnull().values.any())

# Getting rid of the columns with more than 30% missing values. 
missing_fractions = dataset.isnull().mean().sort_values(ascending=False)
missing_fractions.head(10)
drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
dataset.drop(labels=drop_list, axis=1, inplace=True)
dataset.shape

# Given that there are null values drop the rown contianing the null values.
# Fill the missing values with the last value available in the dataset. 
dataset=dataset.fillna(method='ffill')

# Drop the rows containing NA
dataset= dataset.dropna(axis=0)
# Fill na with 0
#dataset.fillna('0')

dataset.tail(2)

# Computing Daily Return
# Daily Log Returns (%)
# datareturns = np.log(data / data.shift(1)) 

# Daily Linear Returns (%)
datareturns = dataset.pct_change(1)

#Remove Outliers beyong 3 standard deviation
datareturns= datareturns[datareturns.apply(lambda x :(x-x.mean()).abs()<(3*x.std()) ).all(1)]

# 3.2. Data Transformation
# All the variables should be on the same scale before applying PCA, otherwise a feature with large values will dominate the result. 
#Below we use StandardScaler in sklearn to standardize the dataset’s features onto unit scale (mean = 0 and variance = 1).
# Standardization is a useful technique to transform attributes to a standard Normal distribution with a mean of
# 0 and a standard deviation of 1.

scaler = StandardScaler().fit(datareturns)
rescaledDataset = pd.DataFrame(scaler.fit_transform(datareturns),columns = datareturns.columns, index = datareturns.index)

# summarize transformed data
datareturns.dropna(how='any', inplace=True)
rescaledDataset.dropna(how='any', inplace=True)
rescaledDataset.head(2)

# 4. Evaluate Algorithms and Models
# 4.1. Train Test Split
# The portfolio is divided into train and test split to perform the analysis regarding the best porfolio
# and backtesting shown later. 

# Dividing the dataset into training and testing sets
percentage = int(len(rescaledDataset) * 0.80)
X_train = rescaledDataset[:percentage]
X_test = rescaledDataset[percentage:]

X_train_raw = datareturns[:percentage]
X_test_raw = datareturns[percentage:]


stock_tickers = rescaledDataset.columns.values
n_tickers = len(stock_tickers)

X_test.head()

# 4.2. Model Evaluation- Applying Principle Component Analysis

pca = PCA()
PrincipalComponent=pca.fit(X_train)

# First Principal Component /Eigenvector
print(pca.components_[0])
print(len(pca.components_))

# 4.2.1.Explained Variance using PCA

NumEigenvalues=10
fig, axes = plt.subplots(ncols=2, figsize=(14,4))
Series1 = pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).sort_values()*100
Series2 = pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).cumsum()*100
Series1.plot.barh(ylim=(0,9), xlabel = "Explained Variance ratio (%)", ylabel="Principal Component",title='Explained Variance Ratio by Top 10 factors',ax=axes[0]);
Series2.plot(ylim=(0,100),xlim=(0,9),ax=axes[1], title='Cumulative Explained Variance by factor', ylabel = "Explained Variance ratio", xlabel = 'Principal Component');
# explained_variance
pd.Series(np.cumsum(pca.explained_variance_ratio_)).to_frame('Explained Variance').head(NumEigenvalues).style.format('{:,.2%}'.format)

# We find that the most important factor explains around 60% of the daily return variation. 
#The dominant factor is usually interpreted as ‘the market’, depending on the results of closer inspection.
# The plot on the right shows the cumulative explained variance and indicates that around 10 factors explain 95% of the returns of this large cross-section of stocks.  

def PCWeights():
    '''
    Principal Components (PC) weights for each 28 PCs
    '''
    weights = pd.DataFrame()

    for i in range(len(pca.components_)):
        weights["weights_{}".format(i)] = pca.components_[i] / sum(pca.components_[i])

    weights = weights.values.T
    return weights

weights=PCWeights()
weights[0]
pca.components_[0]
weights[0]


NumComponents=1
        
topPortfolios = pd.DataFrame(pca.components_[:NumComponents], columns=dataset.columns)
eigen_portfolios = topPortfolios.div(topPortfolios.sum(1), axis=0)
eigen_portfolios.index = [f'Portfolio {i}' for i in range( NumComponents)]
np.sqrt(pca.explained_variance_)
eigen_portfolios.T.plot.bar(subplots=True, layout=(int(NumComponents),1), figsize=(14,10), legend=False, sharey=True, ylim= (-1,1))


# plotting heatmap 
plt.figure(figsize = (15,8))
sns.heatmap(topPortfolios) 

# The heatmap and the plot above shown the contribution of different stocks in each
# eigenvector.
# 4.2.3. Finding the Best Eigen Portfolio
# In order to find the best eigen portfolios and perform backtesting in the next step, we use the sharpe ratio, which is a performance
# metric that explains the annualized returns against the annualized volatility of each company in a portfolio. A high sharpe ratio explains higher returns and/or lower volatility for the specified portfolio. The annualized sharpe ratio is computed by dividing
# the annualized returns against the annualized volatility. For annualized return we
# apply the geometric average of all the returns in respect to the periods per year (days
# of operations in the exchange in a year). Annualized volatility is computed by taking the standard deviation of the returns and multiplying it by the square root of the peri‐
# ods per year.

len(pca.components_)

# Sharpe Ratio
def sharpe_ratio(ts_returns, periods_per_year=252):
    '''
    Sharpe ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk.
    It calculares the annualized return, annualized volatility, and annualized sharpe ratio.
    
    ts_returns are  returns of a signle eigen portfolio.
    '''
    n_years = ts_returns.shape[0]/periods_per_year
    annualized_return = np.power(np.prod(1+ts_returns),(1/n_years))-1
    annualized_vol = ts_returns.std() * np.sqrt(periods_per_year)
    annualized_sharpe = annualized_return / annualized_vol

    return annualized_return, annualized_vol, annualized_sharpe


len(pca.components_)
def optimizedPortfolio():
    n_portfolios = 10
    annualized_ret = np.array([0.] * n_portfolios)
    sharpe_metric = np.array([0.] * n_portfolios)
    annualized_vol = np.array([0.] * n_portfolios)
    highest_sharpe = 0 
    stock_tickers = rescaledDataset.columns.values
    n_tickers = len(stock_tickers)
    pcs = pca.components_
    
    for i in range(n_portfolios):
        
        pc_w = pcs[i] / sum(pcs[i])
        eigen_prtfi = pd.DataFrame(data ={'weights': pc_w.squeeze()*100}, index = stock_tickers)
        eigen_prtfi.sort_values(by=['weights'], ascending=False, inplace=True)
        eigen_prti_returns = np.dot(X_train_raw.loc[:, eigen_prtfi.index], pc_w)
        eigen_prti_returns = pd.Series(eigen_prti_returns.squeeze(), index=X_train_raw.index)
        er, vol, sharpe = sharpe_ratio(eigen_prti_returns)
        annualized_ret[i] = er
        annualized_vol[i] = vol
        sharpe_metric[i] = sharpe
        
        sharpe_metric= np.nan_to_num(sharpe_metric)
        
    # find portfolio with the highest Sharpe ratio
    highest_sharpe = np.argmax(sharpe_metric)

    print('Eigen portfolio #%d with the highest Sharpe. Return %.2f%%, vol = %.2f%%, Sharpe = %.2f' % 
          (highest_sharpe,
           annualized_ret[highest_sharpe]*100, 
           annualized_vol[highest_sharpe]*100, 
           sharpe_metric[highest_sharpe]))


    fig, ax = plt.subplots()
    fig.set_size_inches(12, 4)
    ax.plot(sharpe_metric, linewidth=3)
    ax.set_title('Sharpe ratio of eigen-portfolios')
    ax.set_ylabel('Sharpe ratio')
    ax.set_xlabel('Portfolios')

    results = pd.DataFrame(data={'Return': annualized_ret, 'Vol': annualized_vol, 'Sharpe': sharpe_metric})
    results.dropna(inplace=True)
    results.sort_values(by=['Sharpe'], ascending=False, inplace=True)
    print(results.head(20))

    plt.show()

optimizedPortfolio()

weights = PCWeights()
portfolio = portfolio = pd.DataFrame()

def plotEigen(weights, plot=False, portfolio=portfolio):
    portfolio = pd.DataFrame(data ={'weights': weights.squeeze()*100}, index = stock_tickers) 
    portfolio.sort_values(by=['weights'], ascending=False, inplace=True)
    if plot:
        print('Sum of weights of portfolio represented by first principal component: %.2f' % np.sum(portfolio))
        portfolio.plot(title='Weights of portfolio represented by first principal component', 
            figsize=(12,6), 
            xticks=range(0, len(stock_tickers),1), 
            rot=45, 
            linewidth=3
            )
        plt.show()


    return portfolio

# Weights are stored in arrays, where 0 is the first PC's weights.
plotEigen(weights=weights[0], plot=True)

# The chart shows the allocation of the best portfolio. The weights in the chart are in
# percentages. 

weights[5]

# 4.2.4. Backtesting Eigenportfolio
# We will now try to backtest this algorithm on the test set, by looking at few top and bottom portfolios.

def Backtest(eigen):

    '''

    Plots Principle components returns against real returns.
    
    '''
   
    eigen_prtfi = pd.DataFrame(data ={'weights': eigen.squeeze()}, index = stock_tickers)
    eigen_prtfi.sort_values(by=['weights'], ascending=False, inplace=True)    

    eigen_prti_returns = np.dot(X_test_raw.loc[:, eigen_prtfi.index], eigen)
    eigen_portfolio_returns = pd.Series(eigen_prti_returns.squeeze(), index=X_test_raw.index)
    returns, vol, sharpe = sharpe_ratio(eigen_portfolio_returns)  
    print('Current Eigen-Portfolio:\nReturn = %.2f%%\nVolatility = %.2f%%\nSharpe = %.2f' % (returns*100, vol*100, sharpe))
    equal_weight_return=(X_test_raw * (1/len(pca.components_))).sum(axis=1)    
    df_plot = pd.DataFrame({'PCA Portfolio Return': eigen_portfolio_returns, 'Equal Weight Index': equal_weight_return}, index=X_test.index)
    np.cumprod(df_plot + 1).plot(title='Returns of the equal weighted index vs. PCA portfolio' , 
                          figsize=(12,6), linewidth=3)
    plt.ylabel("Normalized NAV price")
    plt.figure(figsize = (12, 8))
    plt.show()
    
Backtest(eigen=weights[4])
Backtest(eigen=weights[8])
Backtest(eigen=weights[0])

# Checking the performance of first 10 portfolios given by the 10 principal components
for i in range(11):
    Backtest(eigen = weights[i])


Backtest(weights[5])

# Conclusion
# 1. The portfolio given by the first principal component represents a systematic risk factor and 
#    the portfolios given by the other principal component may represent sector or industry factor.
# 2. Since each of the principal component is uncorrelated and independent of each other, 
#    the portfolios given by them can be used for diversification of the portfolio resulting in reduction of risk.

