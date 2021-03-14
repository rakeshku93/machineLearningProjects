# import necessary packages
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import time

def run(fold):
    
    # loading the training dataset
    df = pd.read_csv("../inputs/cat-in-the-dat-train-folds.csv")
    
    # selecting the features on which label_encoding need to be performed 
    features = [ x for x in df.columns if x not in ["id", "target", "kfold"]]
    
    # Filling the NaN values with NONE
    # Converting the values to string, it doesn't impact as all the values are categoricals
    for col in features:
        df.loc[:, col] = df[col].fillna("NONE").astype(str)
        
    for col in features:
        # initiate the label encoder from sklearn
        lbl_enc = preprocessing.LabelEncoder()
        
        lbl_enc.fit(df[col])
        
        df.loc[:, col] = lbl_enc.transform(df[col])   
        
    # get training dataset using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # get validation daatset using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get the training dataset, from dataframe to numpy array
    x_train = df_train[features].values
    
    # get the validation dataset
    x_valid = df_valid[features].values
        
    # Initiate RandomForest Model
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    
    # fit the model on training dataset
    model.fit(x_train, df_train.target.values)
    
    valid_preds = model.predict_proba(x_valid)[:,1]
    
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    
    # print the auc score
    print(f"Fold = {fold}, AUC Score: {auc}")

if __name__ == "__main__":
    start = time.perf_counter()
    for fold in range(5):
        run(fold)
    finish = time.perf_counter()
    print(f"Finished in {finish-start} seconds(s)..")