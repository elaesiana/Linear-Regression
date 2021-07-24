import numpy as np

class LinearRegression():
  """

  Parameters
  X : array_like
    Training data with shape (num_samples, num_features)
  y : array_like
    Target values with shape (num_samples)

  Attributes
  coefficient : array of shape (num_features, 1)
        Estimated coefficients for the linear regression problem.
  intercept : float 
        Independent term in the linear model.
  theta : array of shape (num_features+1,1)
        Intercept and coefficient for the linear regression problem.
    
  """
  def __init__(self, X, y):
  
    self.X = X
    self.y = y

    self.coefficient = None
    self.intercept = None
    self.theta = None
    

  def fit(self):
    """
    Fit linear model.

    Parameters
    X : array_like of shape (num_samples, num_features)
        Training data
    y : array_like of shape (num_samples,)
        Target values. 

    Returns
    self : returns an instance of self.

    """
    self.X = self.X.reshape((self.X.shape[0],1))
    self.y = self.y.reshape(self.y.shape[0],1)

    self.X = np.append(np.ones(self.X.shape), self.X, axis=1)

    m = self.y.shape[0]

    inversed = np.linalg.inv(np.dot(self.X.transpose(),self.X))
    self.theta = inversed.dot(self.X.transpose()).dot(self.y)

    self.coefficient = self.theta[1:]
    self.intercept = self.theta[0][0]

    return self

  def predict(self, X_test):
    """
    Predict class labels for samples in X

    Parameters
    X : array_like of shape (num_samples, num_featues)
      Samples

    Returns
    y_pred : array_like of shape (num_samples)
      Predicted class label per sample.
    """

    X_test = X_test.reshape((X_test.shape[0],1))
    X_test = np.append(np.ones(X_test.shape), X_test, axis=1)
    y_pred = np.dot(X_test,self.theta)

    return y_pred
  
  def r2_score(self,y_true, y_pred):
    """
    R^2 (coefficient of determination) regression score function

    Parameters
    y : array_like of shape (num_samples)
      Ground truth (correct) target values.
    y_pred : array_like of shape (num_samples)
      Estimated target values.

    Returns
    z : float
      The R^2 score

    """
    y_true = y_true.reshape((y_true.shape[0],1))
    y_avg = np.average(y_true)

    return 1-(np.sum(np.power(y_true-y_pred,2))/np.sum(np.power(y_true-y_avg,2)))

  def mean_squared_error(self,y_true,y_pred):
    '''
    Mean squared error regression loss.

    Parameters
    y : array_like of shape (num_samples)
      Ground truth (correct) target values.
    y_pred : array_like of shape (num_samples)
      Estimated target values.

    Returns
    loss : float
      Mean squared error score
    
    '''

    y_true = y_true.reshape((y_true.shape[0],1))
    m = y_true.shape[0]
    
    return 1/m*np.sum(np.power(y_pred-y_true,2))

  def mean_absolute_error(self,y_true,y_pred):
    '''
    Mean absolute error regression loss.

    Parameters
    y : array_like of shape (num_samples)
      Ground truth (correct) target values.
    y_pred : array_like of shape (num_samples)
      Estimated target values.

    Returns
    loss : float
      Mean absolute error score
    
    '''

    y_true = y_true.reshape((y_true.shape[0],1))
    m = y_true.shape[0]

    return 1/m*np.sum(np.abs(y_true-y_pred))