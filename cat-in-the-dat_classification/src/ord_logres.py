import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn import linear_model


def run(fold):

    # loading the training dataset
    df = pd.read_csv("../inputs/cat-in-the-dat-train-folds.csv")

    # Selecting all the categorical columns to perform one-hot encoding
    categorical_features = [
        x for x in df.columns if x not in ["id", "target", "kfold"]
    ]

    # Handling NaN values by filling all NaN values with NONE
    for col in categorical_features:
        # converting all columns data to strings, it doesn't matter
        # as it's categorical dataset < NO Numerical values>
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # get training dataset using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation dataset using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initializing OneHotEncoder from sklearn
    ord_enc = preprocessing.OrdinalEncoder()

    # concatenate training & validation dataset for selected columns
    # fit ohe on training + validation dataset
    full_data = pd.concat(
        [df_train[categorical_features], df_valid[categorical_features]],
        axis=0
    )
    
    ord_enc.fit(full_data)

    # transform training dataset
    # x_train type <class 'scipy.sparse.csr.csr_matrix'>
    x_train = ord_enc.transform(df_train[categorical_features])
    
    # print(df_valid[categorical_features])

    # transform validation dataset
    x_valid = ord_enc.transform(df_valid[categorical_features])

    # initialize Logistic Regression model
    # by default solver = lbfgs, throws error so chnaged to liblinear
    model = linear_model.LogisticRegression()  
    
    # fit the model on training dataset
    model.fit(x_train, df_train.target.values)
    
    # predict on validation data
    # we need probability values to calculate AUC score, 
    # we will be using P(E) of 1s, so [:,1]
    valid_preds = model.predict_proba(x_valid)[:,1] # valid_preds --- [prob of 0, prob of 1]
    
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    
    # print the auc score
    print(f"Fold = {fold}, AUC Score: {auc}")
    
    
if __name__ == "__main__":
    # run the function for all the folds
    for fold in range(5):
        run(fold)
    
    
    
    