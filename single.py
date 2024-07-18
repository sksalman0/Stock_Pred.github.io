import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#pred = les.predict(X_test)

#tom


def model(stock_name):
    data = yf.download(stock_name,period='15y',interval='1d')
    data.drop('Adj Close',axis=1,inplace=True)
    data = data.dropna()
    data[f"{stock_name}_stock"] = data["Close"].shift(-1)
    df_rev = data[['Open','High','Low','Volume']]
    X = df_rev.iloc[:-1,:]
    y= data.iloc[:-1,-1]
    from sklearn.linear_model import Lasso
    les = Lasso()
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)
    les.fit(X_train,y_train)
    X_lat = df_rev.iloc[-1,:]
    X_lat_df = pd.DataFrame(X_lat)
    X_lat_df = X_lat_df.transpose()
    Tom_price = les.predict(X_lat_df)
    Tom_price_str = np.array2string(Tom_price)
    Tom_price_str = Tom_price_str.strip('[]')
    Open = df_rev.iloc[-1,0]
    High = df_rev.iloc[-1,1]
    Low = df_rev.iloc[-1,2]
    Volum = df_rev.iloc[-1,3]
    return Open, High, Low, Tom_price_str

def m_model(stock_name,Open,High,Low,Volum):
    data = yf.download(stock_name,period='15y',interval='1d')
    data.drop('Adj Close',axis=1,inplace=True)
    data = data.dropna()
    data[f"{stock_name}_stock"] = data["Close"].shift(-1)
    df_rev = data[['Open','High','Low','Volume']]
    X = df_rev.iloc[:-1,:]
    y= data.iloc[:-1,-1]
    from sklearn.linear_model import Lasso
    les = Lasso()
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)
    les.fit(X_train,y_train)
    X_lat = np.array([Open,High,Low,Volum]).reshape(1,-1)
    Tom_price = les.predict(X_lat)
    Tom_price_str = np.array2string(Tom_price)
    Tom_price_str = Tom_price_str.strip('[]')
    return Tom_price_str