import pandas as pd
import numpy as np
from sklearn import linear_model
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
    df["income"] = df["income"].str.strip()    # " <=50K" ----> "<=50K"
    
    # map targets to 0s and 1s, .
    target_mapping = {
            "<=50K": 0,
            ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)
    
    # all the categorical features except income & kfold
    features = [x for x in df.columns if x not in ("kfold", "income")]

    # handling NaN values, note that converting all columns to "strings"
    # it doesnt matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
        
    # training dataset
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation dataset
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # ohe full data with the categorical features
    # full_data = df[features]

    # initailize the OneHotEncoding from scikit-learn module
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    ohe.fit(full_data[features])

    # get training data using folds
    x_train = ohe.transform(df_train[features])

    # get validation data using folds
    x_valid = ohe.transform(df_valid[features])

    # initalize LogisticRegression model
    model = linear_model.LogisticRegression()
    
    # check is any NaN values
    # print(np.isnan(df_train.income.values).any())
    
    # fit model on training data
    model.fit(x_train, df_train.income.values)

    # check is any NaN values
    # print(np.isnan(df_train.income.values).any())
    
    yhat_ones = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, yhat_ones)
    
    # print auc at each fold
    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__":
    for fold in range(5):
        run(fold)
    