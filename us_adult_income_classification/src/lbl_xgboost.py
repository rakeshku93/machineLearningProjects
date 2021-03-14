""" dropping numerical features, label encoded xgboost """
import xgboost as xg
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    # load the full training data with folds
    df = pd.read_csv("../inputs/train-folds.csv")

    # list of all numerical columns
    num_cols = [
        "age",
        "fnlwgt",
        "capital-gain",
        "capital-loss",
        "hours-per-week"
    ]
    
    # drop the numerical columns for simplicity
    df = df.drop(num_cols, axis=1)
    
    # remove white-spacing from the values of income column
    df["income"] = df.income.str.strip()

    # map targets to 0s and 1s, .
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)

    # all the categorical features except income & kfold
    features = [x for x in df.columns if x not in ("kfold", "income")]
   
    # initialize LabelEncoder from scikit learn module 
    lbl_enc = preprocessing.LabelEncoder()

    # handling NaN values, note that converting all columns to "strings"
    # it doesnt matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
        
        # fit & transform the data of selected features
        df.loc[:, col] = lbl_enc.fit_transform(df[col])
        
    # training dataset folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # vaidation dataset using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
  
    # get training data
    x_train = df_train[features].values

    # get validation data 
    x_valid = df_valid[features].values

    # initalize xgboost model
    model = xg.XGBClassifier(max_depth=7, n_estimators=200, n_jobs=-1)

    # fit model on training data
    model.fit(x_train, df_train.income.values)

    # predict the probability to get 1s, need to predict 
    # probability values as we are calculating AUC
    yhat_ones = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, yhat_ones)

    # print auc at each fold
    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__":
    for fold in range(5):
        run(fold)

