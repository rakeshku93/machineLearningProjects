""" including numerical features with label encoded xgboost model """
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing


def run(fold):

    df = pd.read_csv("../inputs/train-folds.csv")

    num_cols = [
        "age",
        "fnlwgt",
        "capital-gain",
        "capital-loss",
        "hours-per-week"
    ]

    df["income"] = df.income.str.strip()

    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }

    df.loc[:, "income"] = df.income.map(target_mapping)

    # all columns are features except kfold & income columns
    features = [x for x in df.columns if x not in ("kfold", "income")]

    # initalize labelencoder from scikit-learn module 
    lbl_enc = preprocessing.LabelEncoder()

    # fill all NaN values with NONE, converting all columns to "strings"
    # it doesnt matter because all are categories
    for col in features:
        # Label Encoding the categorical features, leaving numerical fetaures untouched!
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

            df.loc[:, col] = lbl_enc.fit_transform(df[col])

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data, converting to numpy array
    x_train = df_train[features].values

    # get validation data
    x_valid = df_valid[features].values

    # initalize xgboost model
    model = xgb.XGBClassifier()
    
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
