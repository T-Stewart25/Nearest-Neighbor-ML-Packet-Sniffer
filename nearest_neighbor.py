# Description: This file contains the implementation of the nearest neighbor classifier.
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold

def nearest(X_train, y_train, X_test, y_test):
    #Create 50 stratified k-folds with random shuffling, mainly used to randomize data more since already split and shuffled
    skf = StratifiedKFold(n_splits=50, random_state=42, shuffle=True)
    #Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    #Create KNN classifier
    knn = KNeighborsClassifier(10)
    for train_index, train2_index in skf.split(X_train, y_train):
        X_train_fold, X_train2_fold = X_train[train_index], X_train[train2_index] #Get the training data
        y_train_fold, y_train2_fold = y_train[train_index], y_train[train2_index] #Get more training data
        knn.fit(X_train_fold, y_train_fold)#Fit the model
        knn.fit(X_train2_fold, y_train2_fold)#Fit the model again. Since data is already split for validation, we can fit the model again

    y_pred = knn.predict(X_test) #Predict the test data
   
    return y_pred
    
def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)