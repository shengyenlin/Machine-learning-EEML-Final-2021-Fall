import numpy as np
import pandas as pd

import importlib
import TechnicalAnalysisFeatures
importlib.reload(TechnicalAnalysisFeatures)
from TechnicalAnalysisFeatures import *

ASSET_IDX = 1
LOAD_STRICT = 0

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

    #######feature engineering#######
    #parameters for N (lookback period)
    Ns = [3, 5, 10, 30]

    #parameters for KST
    r1, r2, r3, r4 = 10, 15, 20, 30
    n1, n2, n3, n4 = 10, 10, 10, 15

    #parameter for MACD 
    n_fasts = [3, 5]
    n_slows = [10, 15]

    #parameter for ADX
    n_ADX = 10

    #features with parameter n
    for n in Ns:
        train = MA(train, n)
        train = EMA(train, n)
        train = MOM(train, n)
        train = ATR(train, n)
        train = BBANDS(train, n)
        train = TRIX(train, n)
        train = ADX(train, n, n_ADX)
        train = Vortex(train, n)
        train = RSI(train, n)
        train = ACCDIST(train, n)
        train = MFI(train, n)
        train = OBV(train, n)
        train = FORCE(train, n)
        train = EOM(train, n)
        train = CCI(train, n)
        train = COPP(train, n)
        train = KELCH(train, n)
        train = DONCH(train, n)
        train = STDDEV(train, n)

        #MACD
        for n_fast in n_fasts:
            for n_slow in n_slows:
                train = MACD(train, n_fast, n_slow)

        #feature without parameter n
        train = PPSR(train)
        train = STOK(train)
        train = MassI(train)
        train = Chaikin(train)
        train = ULTOSC(train)

        #KST
        train = KST(train, r1, r2, r3, r4, n1, n2, n3, n4)