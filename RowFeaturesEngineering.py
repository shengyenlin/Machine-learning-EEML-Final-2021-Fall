import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

# Two features from the competition tutorial
def upper_shadow(df): return df['High'] - np.maximum(df['Close'], df['Open'])
def lower_shadow(df): return np.minimum(df['Close'], df['Open']) - df['Low']

# A utility function to build features from the original df
def get_features(df):
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target']].copy()

    df_feat["high_div_low"] = df_feat["High"] / df_feat["Low"]
    df_feat["open_sub_close"] = df_feat["Open"] - df_feat["Close"]

    df_feat['trade']=df_feat['Close']-df_feat['Open']
    df_feat['gtrade']=df_feat['trade']/df_feat['Count']

    df_feat['upper_Shadow'] = upper_shadow(df_feat)
    df_feat['lower_Shadow'] = lower_shadow(df_feat)   
    df_feat['upper_Shadow_log']=np.log(df_feat['upper_Shadow'])
    df_feat['lower_Shadow_log']=np.log(df_feat['lower_Shadow'])
    df_feat['shadow1']=df_feat['trade']/df_feat['Volume']
    df_feat['shadow2']=df_feat['upper_Shadow']/df['Low']
    df_feat['shadow3']=df_feat['upper_Shadow']/df['Volume']
    df_feat['shadow4']=df_feat['lower_Shadow']/df['High']
    df_feat['shadow5']=df_feat['lower_Shadow']/df['Volume']    

    df_feat['spread'] = df_feat['High'] - df_feat['Low']
    df_feat['mean_trade'] = df_feat['Volume']/df_feat['Count']
    df_feat['log_price_change'] = np.log(df_feat['Close']/df_feat['Open'])
    df_feat['diff1'] = df_feat['Volume'] - df_feat['Count']
    df_feat['mean1'] = (df_feat['shadow5'] + df_feat['shadow3']) / 2
    df_feat['mean2'] = (df_feat['shadow1'] + df_feat['Volume']) / 2
    df_feat['mean3'] = (df_feat['trade'] + df_feat['gtrade']) / 2
    df_feat['mean4'] = (df_feat['diff1'] + df_feat['upper_Shadow']) / 2
    df_feat['mean5'] = (df_feat['diff1'] + df_feat['lower_Shadow']) / 2
    df_feat['UPS'] = (df_feat['High'] - np.maximum(df_feat['Close'], df_feat['Open']))
    df_feat['UPS'] = df_feat['UPS']
    df_feat['LOS'] = (np.minimum(df_feat['Close'], df_feat['Open']) - df_feat['Low'])
    df_feat['LOS'] = df_feat['LOS']
    df_feat['RNG'] = ((df_feat['High'] - df_feat['Low']) / df_feat['VWAP'])
    df_feat['RNG'] = df_feat['RNG']
    df_feat['MOV'] = ((df_feat['Close'] - df_feat['Open']) / df_feat['VWAP'])
    df_feat['MOV'] = df_feat['MOV']
    df_feat['CLS'] = ((df_feat['Close'] - df_feat['VWAP']) / df_feat['VWAP'])
    df_feat['CLS'] = df_feat['CLS']
    df_feat['LOGVOL'] = np.log(1. + df_feat['Volume'])
    df_feat['LOGVOL'] = df_feat['LOGVOL']
    df_feat['LOGCNT'] = np.log(1. + df_feat['Count'])
    df_feat['LOGCNT'] = df_feat['LOGCNT']
    df_feat["Close/Open"] = df_feat["Close"] / df_feat["Open"]
    df_feat["Close-Open"] = df_feat["Close"] - df_feat["Open"]
    df_feat["High-Low"] = df_feat["High"] - df_feat["Low"]
    df_feat["High/Low"] = df_feat["High"] / df_feat["Low"]
    df_feat['Mean'] = df_feat[['Open', 'High', 'Low', 'Close']].mean(axis = 1)
    df_feat["High/Mean"] = df_feat["High"] / df_feat["Mean"]
    df_feat["Low/Mean"] = df_feat["Low"] / df_feat["Mean"]
    df_feat["Volume/Count"] = df_feat["Volume"] / (df_feat["Count"] + 1)
    mean_price = df_feat[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    median_price = df_feat[['Open', 'High', 'Low', 'Close']].median(axis=1)
    df_feat['high2mean'] = df_feat['High'] / mean_price
    df_feat['low2mean'] = df_feat['Low'] / mean_price
    df_feat['high2median'] = df_feat['High'] / median_price
    df_feat['low2median'] = df_feat['Low'] / median_price
    df_feat['volume2count'] = df_feat['Volume'] / (df_feat['Count'] + 1)
    return df_feat

ASSET_IDX = 1
LOAD_STRICT = 0
SEED = 1126
THRESHOLD = 0.05

if __name__ == "__main__":
    data_path = './data/'
    train = pd.read_csv(
        data_path + 'train.csv'
        #,nrows = 1000
        )

    #preprocessing for timestep
    train['date'] = pd.to_datetime(train['timestamp'], unit = 's')
    train = train.sort_values('date')
    train.set_index(train['timestamp'], inplace=True)
    train.drop(['timestamp'], axis = 1, inplace=True)
    
    #Whether to load strict data or not
    if LOAD_STRICT: 
        train = train.loc[train['date'] < "2021-06-13 00:00:00"]

    #Choose which asset
    train = train[train["Asset_ID"] == ASSET_IDX]

    #fill missing timestamp
    train = train.reindex(
        range(train.index[0],train.index[-1]+60,60), method='pad'
    )

    #feature engineering
    train = get_features(train)

    #feature selection
    drop_cols = ["Target", "Asset_ID", "date"]
    keep_cols = list(
        set(train.columns) - set(drop_cols)
    )
    y_train = train["Target"]
    X_train = train.loc[:, keep_cols]

    #feature selection by L1 regreesion
    #Lasso can't handle NaN
    X_train_ = X_train.fillna(0)
    clf = LassoCV(
        random_state=SEED, max_iter = 50000, cv = 5
        ).fit(X_train_, y_train)

    final_features_l1Reg = X_train.columns[
        (abs(clf.coef_) > 0).flatten()
    ]
    print(f"There are {len(final_features_l1Reg)} features selected by L1 regression")
    print(final_features_l1Reg.tolist())

    #feature selection by pearson correlation
    #fill nan with mean to avoid bias
    X_train_ = X_train.fillna(X_train.mean())
    corr_value = X_train_.corrwith(y_train)
    corr_abs = abs(corr_value).sort_values(ascending=False)
    final_features_corr = corr_abs[corr_abs > THRESHOLD].index
    print(f"There are {len(final_features_corr)} features selected by correlation coefficient")
    print(final_features_corr.tolist())

    #Union final features
    final_features = list(
        set(final_features_l1Reg).union(
            set(final_features_corr)
        )
    )
    
    X_train = X_train[final_features]