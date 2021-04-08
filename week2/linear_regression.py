import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))


class LinearRegression(object):
    def __init__(self):
        pass

    # Calculating the cost function
    def computecost(self, x, y, theta):
        # Calculate the squared term
        inner = np.power(((x * theta.T) - y), 2)
        return np.sum(inner) / (2 * x.shape[0])

    # Gradient descent, the incoming parameters are matrices
    def gradient_descent(self, X, y, alpha, iters):
        theta = np.mat(np.zeros(X.shape[1]))
        temp = np.matrix(np.zeros(theta.shape))
        # Number of features
        parameters = int(theta.ravel().shape[1])

        # iters sub iterations
        for i in range(iters):
            error = (X * theta.T) - y
            for j in range(parameters):
                # Record the intermediate term
                term = np.multiply(error, X[:, j])
                temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

            theta = temp
        return theta

    # Least squares method with incoming parameters as matrix
    def normaleqn(self, X, y):
        # X.T@X equivalent to X.T.dot(X)
        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        return theta.T

    # Calculate the mean square error
    def MSE(self, x, y, theta):
        inner = np.power(((x * theta.T) - y), 2)
        return np.sum(inner) / (x.shape[0])

    # Calculate the root mean square error
    def RMSE(self, x, y, theta):
        return np.sqrt(self.MSE(x, y, theta))

    # Data Visualization
    def visualization(self,x,y):
        pass


if __name__ == '__main__':
    # Data pre-processing
    data = load_boston()
    X = pd.DataFrame(data.data)
    y = data.target
    X.columns = data.feature_names
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
    Xtrain, Xtest = pd.DataFrame(Xtrain.loc[:, 'LSTAT']), pd.DataFrame(Xtest.loc[:, 'LSTAT'])
    Xtrain.insert(0, 'Ones', 1)
    Xtrain = np.mat(Xtrain)
    Xtest.insert(0, 'Ones', 1)
    Xtest = np.mat(Xtest)
    Ytrain, Ytest = np.mat(Ytrain).T, np.mat(Ytest).T

    # Model Selection
    LN = LinearRegression()
    theta1 = LN.gradient_descent(Xtrain, Ytrain, 0.0093, 1000)
    print('The Mean Square Error of gradient descent is {:.5}'.format(LN.MSE(Xtest, Ytest, theta1)))
    theta2 = LN.normaleqn(Xtrain, Ytrain)
    print('The Mean Square Error of least squares method is {:.5}'.format(LN.MSE(Xtest, Ytest, theta2)))

    # Data Visualization
    x = np.linspace(Xtrain[:, 1:].min(), Xtrain[:, 1:].max(), 100)
    g_x = theta1[0, 0] + (theta1[0, 1] * x)
    f_x = theta2[0, 0] + (theta2[0, 1] * x)

    plt.subplots(figsize=(12, 8))
    plt.plot(x, f_x, 'g', label='Prediction1')
    plt.plot(x, g_x, 'b', label='Prediction2')
    plt.scatter(X['LSTAT'], y, label='Data')
    plt.legend(loc=1)
    plt.xlabel('LSTAT', fontsize=18)
    plt.ylabel('Target', fontsize=18)
    plt.title('Gradient Descent vs. Least Squares Method', fontsize=20)
    plt.show()
