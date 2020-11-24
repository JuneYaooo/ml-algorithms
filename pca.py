import numpy as np
import scipy as sp
import pandas as pd

class PCA(object):
    """
    Principal component analysis (PCA).
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.
    Parameters
    ----------
    n_component_ratio: select the number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_component_ratio
    ----------
    
    function1-fit_transform: Fit the model by computing full SVD on X.
    Parameters
    ----------
    X: x_train
    ----------
    
    function2-transform: Fit the model with X and apply the dimensionality reduction on y.
    Parameters
    ----------
    y : x_test
    ----------
    """
    
    def __init__(self,n_component_ratio):
        self.n_component_ratio=n_component_ratio
        
    def fit_transform(self,x):
        n_samples, n_features = x.shape       
        X_center = x-np.mean(x, axis=0)   # Center data
        U, s, Vt=sp.linalg.svd(X_center, full_matrices=False)
        self.__Vt=Vt
        explained_variance = (s ** 2) / (n_samples - 1)   # Get variance explained by singular values
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var   #caculate the compressed rate
        self.n_components=self.__choose_ratio(explained_r=explained_variance_ratio)  #find how many features we should keep on the compressed rate we select
        x_compressed = U[:, :self.n_components].dot(np.diag(s[:self.n_components]))  #return the features we choose
        del X_center,explained_variance,U,s,Vt      #del variable and release memory
        self.explained_variance_ratio=explained_variance_ratio[:self.n_components] #make explained variance ratio the attributes in pca, so that we can print it out
        return x_compressed
    
    def __choose_ratio(self,explained_r):
        for i in range(1, len(explained_r)):
            if sum(explained_r[:i])/sum(explained_r) >= self.n_component_ratio:
                return i
            
    def transform(self,y):
        y_centre = y-np.mean(y, axis=0)
        y_compressed=y_centre.dot(np.linalg.inv(self.__Vt)[:,:self.n_components])   # compress x_test based on the Vt on x_train
        del y_centre  #del variable and release memory
        gc.collect()
        return y_compressed.values