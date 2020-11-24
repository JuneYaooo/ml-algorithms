import numpy as np
import pandas as pd

class KNN(object):
    """
    Classifier implementing the k-nearest neighbors vote.
    Parameters
    ----------
    n_neighbors: int, required, default=None
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    ----------
    
    function1-fit: Fit the model by k-nearest neighbors.
    Parameters
    ----------
    X_train: train dataset
    y_train: label of X_train
    ----------
    
    function2-predict: return x_test's classification result after KNN process.
    Parameters
    ----------
    y : x_test
    ----------
    """
    
    def __init__(self,n_neighbors):
        self.n_neighbors=n_neighbors
        self._X_train = None 
        self._y_train = None
        
    def fit(self,X_train,y_train):
        self._X_train=X_train
        self._y_train=y_train
        return self 
            
    def predict(self,X_test):
        distances = [np.linalg.norm(x_test-self._X_train,ord=2,axis=1) for x_test in X_test.values] # caculate the distance
        distances_sort=[ np.argsort(distance)[0:self.n_neighbors] for distance in distances] # sort by distance and only select Top n_neighbors point with shortest distance
        target=[self._y_train[distance_s] for distance_s in distances_sort]  # get the label of these point
        class_result=sp.stats.mode(target, axis = 1)[0].flatten()  #select mode label of the distance
        del distances,distances_sort,target   #del variable and release memory
        return class_result