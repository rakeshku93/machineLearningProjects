import pandas as pd
import numpy as np
import time
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn import ensemble


def run(fold):
    """
    Here tried to implement Label Encoder all data at once, which throws error.
    BECAUSE, LabelEncoder to be performed for one categorical column at a time, For same implemetation like 
    Label Encoder & all data at once., We can use Ordinal Encoder.
    
    Conclusion:
    Label Encoder to be performed for one categorical column at a time.
    While, One Hot Encoding performed for all whole categorical data at once.
    """

    df = pd.read_csv("../inputs/cat-in-the-dat-train-folds.csv")
    
    # # extracting the categorical features
    cat_features = [x for x in df.columns if x not in ("id", "target", "kfold")]
    
    for col in cat_features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ord_enc = preprocessing.OrdinalEncoder()

    full_cat_data = pd.concat(
        [df_train[cat_features], df_valid[cat_features]],
        axis=0
    )

    ord_enc.fit(full_cat_data)

    x_train = ord_enc.transform(df_train[cat_features])

    x_valid = ord_enc.transform(df_valid[cat_features])

    model = ensemble.RandomForestClassifier(n_jobs=-1)

    model.fit(x_train, df_train.target.values)

    yhat_ones = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.target.values, yhat_ones)

    # print the auc score
    print(f"Fold = {fold}, AUC Score: {auc}")


if __name__ == "__main__":
    start = time.perf_counter()
    for fold in range(5):
        run(fold)
    finish = time.perf_counter()
    print(f"Finished in {finish-start} seconds(s)..")


