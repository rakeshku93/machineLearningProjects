# import packages
import pandas as pd
from sklearn import model_selection

def create_folds(path):
    # read the csv file
    data = pd.read_csv(path)
    
    # create a new column "kfold" and fill with -1
    data["kfold"] = -1
    
    # randomized the dataframe rows
    data = data.sample(frac=1).reset_index(drop=True)
    
    # fetch labels
    y = data.income.values
    
    # initalized the stratifiedKFold class from model_selection module
    stratified_kf = model_selection.StratifiedKFold(n_splits=5, random_state=0)
    
    # for fold, (train_index, test_index) in stratified_kf.split(X_train, y_train):
    # fill the new kfold column, specific rows corresponding to the test_data with fold values
    for fold, (t_, v_) in enumerate(stratified_kf.split(X=data, y=y)):
        data.loc[v_, "kfold"] = fold
    
    # save the new csv file with kfold column   
    return data.to_csv("../inputs/train-folds.csv", 
                       index=False) 
    
if __name__ == "__main__":
    path = "../inputs/train.csv"
    create_folds(path)





