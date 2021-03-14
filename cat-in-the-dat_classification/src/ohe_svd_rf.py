"""
 One Hot Encoding with Singular value composition -- Random Forest
"""
# import necessary packages
import pandas as pd
from scipy import sparse
from sklearn import decomposition
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import ensemble

def run(fold):
    # load the training data with folds
    df = pd.read_csv("../inputs/cat-in-the-dat-train-folds.csv")

    # extracting the categorical features
    features = [ x for x in df.columns if x not in ("id", "target", "kfold")]
    
    # Handling NaN values by replacing with NONE
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
        
    # training dataset
    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    
    # validation dataset
    df_valid = df[df["kfold"] == fold].reset_index(drop=True)
    
    # full_data = pd.concat(
    #         [df_train[features], df_valid[features]], 
    #         axis=0
    # ) 
    
    full_data = df[features]
    
    # initalize the OneHotEncoder() from sklearn
    ohe = preprocessing.OneHotEncoder()
    
    # fit to the data
    ohe.fit(full_data[features])
    
    # transform training dataset
    x_train = ohe.transform(df_train[features])
     
    # transform validation dataset
    x_valid = ohe.transform(df_valid[features])
    
    # initialize Truncated SVD
    # we are reducing the data to 120 components
    svd = decomposition.TruncatedSVD(n_components=120)
    
    # fit svd on full sparse training data
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)
    
    # transform the training sparse data    
    x_train = svd.transform(x_train)
    
    # transform the validation sparse data
    x_valid = svd.transform(x_valid)
    
    # initialize the RandomForestClassifier
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    
    # fit the data to the model
    model.fit(x_train, df_train.target.values)
    
    # predict only the ones for the given x_valid dataset,
    # need to select ones so [:,1] -- all rows of 2nd column, 1st column for zeros 
    yhat_ones = model.predict_proba(x_valid)[:,1]
        
    # evaluate the auc score
    auc = metrics.roc_auc_score(df_valid.target.values, yhat_ones)
    
    print(f"Fold: {fold}, AUC Score: {auc}")
    
if __name__ == "__main__":
    for fold in range(5):
        run(fold)