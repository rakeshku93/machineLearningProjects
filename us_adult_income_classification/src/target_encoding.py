import copy
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import xgboost as xgb


def mean_target_encoding(data):
    
    df = copy.deepcopy(data)
    
    # numerical columns
    num_cols = [
        "age",
        "fnlwgt",
        "capital-gain",
        "capital-loss",
        "hours-per-week"
    ]
    
    # removing whitespacing from income column
    df.loc[:, "income"] = df.income.str.strip()
    
    # mapping the income to 0s & 1s 
    target_mapping = {
        "<=50K" : 0,
        ">50K" : 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)
    
    # all the categorical features, No num_cols , kfold & income columns   
    features = [x for x in df.columns if x not in num_cols and x not in ("kfold", "income")]
    
    
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
        
    for col in features:
        lbl_enc = preprocessing.LabelEncoder()
        
        lbl_enc.fit(df[col])
        
        df.loc[:, col] = lbl_enc.transform(df[col])  
        
    encoded_dfs = []
    
    for fold in range(5):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        
        for col in features:
            mapping_dict = dict(
                df_train.groupby(col)["income"].mean()
            )
            
            df_valid.loc[:, col + "_enc"] = df_valid[col].map(mapping_dict)
            
        encoded_dfs.append(df_valid)
        
    encoded_df = pd.concat(encoded_dfs, axis=0)
    
    # print(encoded_df.columns)
    
    return encoded_df

def run(df, fold):
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    df_valid = df[df.kfold == fold_].reset_index(drop=True)
    
    features = [x for x in df.columns if x not in ("kfold", "income")]
    
    x_train = df_train[features].values
    
    x_valid = df_valid[features].values
    
    # initialize xgboost model
    model = xgb.XGBClassifier(n_jobs=-1, max_depth=7)
    
    # fit model on training data (ohe)
    model.fit(x_train, df_train.income.values)
    
    # predict on validation data, need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    
    # print auc
    print(f"Fold = {fold}, AUC = {auc}")
    
if __name__ == "__main__":
    data = pd.read_csv("../inputs/train-folds.csv")
    
    df = mean_target_encoding(data)
    
    for fold_ in range(5):
        run(df, fold_)
    