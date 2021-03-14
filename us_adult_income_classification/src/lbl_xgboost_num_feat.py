import itertools
from timeit import main
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing


def feature_engineering(df, cat_cols):
    
    # this will create all 2-combinations of values in this list
    # for example: list(itertools.combinations([1, 2, 3, 9], 2)) will return
    # [(1, 2), (1, 3), (1,9), (2, 3), (2,9), (3,9)]
    combi = itertools.combinations(cat_cols, 2)
    
    for c1, c2 in combi:
        # Merging of two features
        # create a new column named c1_c2 in the dataframe i.e race_sex, sex_nativecountry etc.
        df.loc[:, c1 + "_" + c2] = df[c1].astype(str) + "_" + df[c2].astype(str)
        
    return df

def run(fold):
    # loading training data with folds
    df = pd.read_csv("../inputs/train-folds.csv")
    
    # removing whitespacing from the income column
    df["income"] = df.income.str.strip()
    
    # mapping the income to 0s & 1s 
    target_mapping = {
        "<=50K" : 0,
        ">50K" : 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)
    
    # alternate method---
    # df.income.apply(lambda x: 0 if x < "50K" else 0)
    
    # numerical columns
    num_cols = [
        "age",
        "fnlwgt",
        "capital-gain",
        "capital-loss",
        "hours-per-week"
    ]
    
    # list of only cateogrical features 
    cat_cols = [x for x in df.columns if x not in num_cols and x not in ("kfold", "income")]
    
    df = feature_engineering(df, cat_cols)   
    
    # all features i.e. features along with the combination of features, except kfold & income
    features = [ x for x in df.columns if x not in ("kfold", "income")]
    
    lbl_enc = preprocessing.LabelEncoder()
    
    # handing NaN values for categorical features
    for col in features:
        if col not in num_cols:
            # fill the NaN values with NONE, converted all columns to string
            # it doesnot matter because all are categories 
            df.loc[:, col] = df[col].astype(str).fillna("NONE")
            
            # fit & transform the selected feature
            df.loc[:, col] = lbl_enc.fit_transform(df[col])
            
    # training data with folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # validation data with folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # get training data
    x_train = df_train[features].values
    
    # get validation data
    x_valid = df_valid[features].values
    
    # initialize the xgboost model
    model = xgb.XGBClassifier()
    
    # fit the model on training data
    model.fit(x_train, df_train.income.values)
    
    # predict on validation data, need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]
    
    # auc score
    auc= metrics.roc_auc_score(df_valid.income.values, valid_preds)
    
    # print the auc score for each fold
    print(f"Fold = {fold}, AUC = {auc}")
    
if __name__ == "__main__":
    for fold in range(5):
        run(fold)
            
            
            
            
    
        
        
        
