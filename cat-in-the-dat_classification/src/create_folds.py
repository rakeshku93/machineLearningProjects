# import packages
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import manifold

def create_folds(path):
    # read the csv file
    data = pd.read_csv(path)
    
    # create a new column "kfold" and fill with -1
    data["kfold"] = -1
    
    # randomized the dataframe rows
    data = data.sample(frac=1).reset_index(drop=True)
    
    # fetch labels
    y = data.target.values
    
    # initalized the stratifiedKFold class from model_selection module
    stratified_kf = model_selection.StratifiedKFold(n_splits=5, random_state=0)
    
    # fill the new kfold column specific rows with fold values
    for fold, (t_, v_) in enumerate(stratified_kf.split(X=data, y=data.target.values)):
        data.loc[v_, "kfold"] = fold
    
    # save the new csv file with kfold column   
    return data.to_csv("../inputs/cat-in-the-dat-train-folds.csv", 
                       index=False) 
    
if __name__ == "__main__":
    path = "../inputs/cat-in-the-dat-train.csv"
    create_folds(path)





