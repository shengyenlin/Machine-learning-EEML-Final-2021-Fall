import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV

SEED = 1126
DATA_PATH = "./data_with_features.csv"
THRESHOLD = 0.05 #threshold for selecting features determined by correlation coefficient

if __name__ == "__main__":
    train = pd.read_csv(DATA_PATH)
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
        cv=5,random_state=SEED, max_iter = 50000
        ).fit(X_train_, y_train)

    acc = clf.score(X_train_, y_train)
    print(f"Training acc of best model = {round(acc * 100, 4)}%")
    print("Cross Validation result in each parameter sets")
    pd.DataFrame({"Alpha":clf.alphas_, "Cross valid MSE": clf.mse_path_.mean(axis = 1)})

    final_features_l1Reg = X_train.columns[(clf.coef_ > 0).flatten()]
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
