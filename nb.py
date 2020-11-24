import numpy as np
import pandas as pd

class Gaussian(object):
    """
    implements Gaussian Bayes classifier algorithms. These are supervised learning methods based on applying Bayes' theorem with strong (naive) feature independence assumptions.
    
    function1-fit: Fit the model by Gaussian Bayes classifier.
    Parameters
    ----------
    X_train: train dataset
    y_train: label of X_train
    ----------
    
    function2-predict: return x_test's classification result based on Gaussian Bayes classifier.
    Parameters
    ----------
    y : x_test
    ----------
    """
    
    def __init__(self):
        pass
        
    def fit(self,X_train,y_train):
        self._data_with_label=X_train.copy()
        self._y_train=pd.DataFrame(y_train.values,columns=['label'])
        self._data_with_label['label']=y_train.values
        self.mean_mat= self._data_with_label.groupby("label").mean()
        self.var_mat=self._data_with_label.groupby("label").var()
        self.prior_rate=self.__Priori()
        return self
            
    def predict(self,X_test):
        pred=[ self.__Condition_formula(self.mean_mat,self.var_mat,row )*self.prior_rate for row in X_test.values ]  # get the 
        class_result=np.argmax(pred, axis=1)  # get the max 
        return class_result
    
    #Priori probability
    def __Priori(self):
        la = self._y_train['label'].value_counts().sort_index()
        prior_rate=np.array([ i /sum(la) for i in la])
        return prior_rate
    
    #Gaussian Bayes condition formula
    def __Condition_formula(self,mu,sigma2,row):
        P_mat=1/np.sqrt(2*math.pi*sigma2)*np.exp(-(row-mu)**2/(2*sigma2))
        P_mat=pd.DataFrame(P_mat).prod(axis=1)
        return P_mat