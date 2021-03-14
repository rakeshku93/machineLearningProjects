""" Label Encoder with XGBoost"""

import xgboost as xg
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics


def run(fold):
    
    df = pd.read_csv("../inputs/cat-in-the-dat-train-folds.csv")
    
    features = [x for x in df.columns if x not in ("id", "target", "kfold")]
    
    lbl_enc = preprocessing.LabelEncoder()
    
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
        
        lbl_enc.fit(df[col])
        
        df.loc[:, col] = lbl_enc.transform(df[col])
        
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    x_train = df_train[features]
    
    x_valid = df_valid[features]
    
    model = xg.XGBClassifier()
    
    model.fit(x_train, df_train.target.values)
    
    # predict on validation data we need the probability 
    # values as we are calculating AUC, we will use the probability of 1s
    yhat_ones = model.predict_proba(x_valid)[:,1]
    
    auc = metrics.roc_auc_score(df_valid.target.values, yhat_ones)
    
    print(f"Fold = {fold}, AUC = {auc}")
    
if __name__ == "__main__":
    for fold in range(5):
        run(fold)
    
        
    
    