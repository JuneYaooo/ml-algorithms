class Tool(object):
    """
    function1-train_test_split: Split arrays or matrices into random train and test subsets
    Parameters
    ----------
    X: train_data before split
    y: test_data before split
    test_size : float or int, default=None
        Should be float, and should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. default is 0.25
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
    ----------
    
    function2-accuracy_score: Accuracy classification score.
    Parameters
    ----------
    y : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    ----------
    """
    
    def train_test_split(self,X, y, test_size=0.25,  random_state=None):
        if random_state:
            np.random.seed(random_state)
        idx=np.random.permutation(X.shape[0])
        test_sz=int(test_size*X.shape[0])
        x_train, x_test = X.iloc[idx][test_sz:], X.iloc[idx][:test_sz]
        y_train, y_test = y.iloc[idx][test_sz:], y.iloc[idx][:test_sz]
        return x_train, x_test, y_train, y_test

    def accuracy_score(self,y, y_pred):
        return float(sum(yi == yi_pred for yi, yi_pred in zip(np.array(y), np.array(y_pred))) / len(y))