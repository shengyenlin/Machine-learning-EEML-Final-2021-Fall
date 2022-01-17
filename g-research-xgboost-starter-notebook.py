#!/usr/bin/env python
# coding: utf-8

# In[2]:


import traceback
import numpy as np
import pandas as pd
import datatable as dt
import gresearch_crypto
from lightgbm import LGBMRegressor

TRAIN_JAY = '../input/cryptocurrency-extra-data-binance-coin/orig_train.jay'
ASSET_DETAILS_JAY = '../input/cryptocurrency-extra-data-binance-coin/orig_asset_details.jay'


# In[3]:


df_train = dt.fread('../input/cryptocurrency-extra-data-binance-coin/orig_train.jay').to_pandas()
df_train.head()


# In[4]:


df_asset_details = dt.fread('../input/cryptocurrency-extra-data-binance-coin/orig_asset_details.jay').to_pandas().sort_values("Asset_ID")
df_asset_details


# # Training

# ## Feature Extraction

# In[5]:


def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

def get_features(df):
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    return df_feat


# ## Feature Selection

# In[6]:


from sklearn.linear_model import LassoCV

SEED = 1126
THRESHOLD = 0.0015
def feature_selection_L1Regression(X_train, y_train):
    clf = LassoCV(
        cv=5,random_state=SEED, max_iter = 50000
        ).fit(X_train, y_train)

    acc = clf.score(X_train, y_train)
    print(f"Training acc of best model = {round(acc * 100, 4)}%")
    #print(f"Cross Validation result in each parameter sets: {clf.coef}")
    pd.DataFrame({"Alpha":clf.alphas_, "Cross valid MSE": clf.mse_path_.mean(axis = 1)})
    
    #final_features_l1Reg = X_train.columns[(clf.coef_ > 0).flatten()]
    print(f"There are {len(final_features_l1Reg)} features selected by L1 regression")
    print(final_features_l1Reg.tolist())
    final_features_l1Reg = final_features_l1Reg[:0.8 * len(final_features_l1Reg)]
    return final_features_l1Reg
    
def feature_selection_PearsonCorrelation(X_train, y_train):
    #feature selection by pearson correlation
    corr_value = X_train.corrwith(y_train)
    corr_abs = abs(corr_value).sort_values(ascending=False)
    final_features_corr = corr_abs[corr_abs > THRESHOLD].index
    print(f"There are {len(final_features_corr)} features selected by correlation coefficient: {corr_abs}")
    print(final_features_corr.tolist())

    return final_features_corr


# ## Main Training Function

# In[13]:


import xgboost as xgb
def get_Xy_and_model_for_asset(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    df_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_proc = df_proc.dropna(how="any")
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    
    #X = X[feature_selection_PearsonCorrelation(X, y)]
    #print(X.head())
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=11,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.7,
        missing=-999,
        random_state=2020,
    )
    model.fit(X, y)

    return X, y, model


# ## Loop over all assets

# In[12]:


df_asset_details


# In[15]:


Xs = {}
ys = {}
models = {}

for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
    try:
        X, y, model = get_Xy_and_model_for_asset(df_train, asset_id)    
        Xs[asset_id], ys[asset_id], models[asset_id] = X, y, model
    except:         
        Xs[asset_id], ys[asset_id], models[asset_id] = None, None, None    


# # Evaluation

# In[17]:


from sklearn.metrics import mean_squared_error

def getScore(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    df_proc = df_proc.dropna(how="any")
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    predicted = models[asset_id].predict(X)
    return mean_squared_error(predicted, y)

for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    mse = getScore(df_train, asset_id)
    print(f"MSE for {asset_name:<16} (ID={asset_id:<2}) = {mse}")


# # Submit To Kaggle

# In[ ]:


env = gresearch_crypto.make_env()
iter_test = env.iter_test()

for i, (df_test, df_pred) in enumerate(iter_test):
    for j , row in df_test.iterrows():
        
        if models[row['Asset_ID']] is not None:
            try:
                model = models[row['Asset_ID']]
                x_test = get_features(row)
                y_pred = model.predict(pd.DataFrame([x_test]))[0]
                df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = y_pred
            except:
                df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = 0
                traceback.print_exc()
        else: 
            df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = 0
        
    env.predict(df_pred)

