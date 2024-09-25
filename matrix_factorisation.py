import numpy as np
import metric

class matrix_factorisation:
    """
    An object of this class will implement the training and
    """
    def __init__(self, k, m,n):
        """
        The object will have the I matrix and the U matrix.
        The training will use some methods to
        """
        self.k = k     # TODO: Est ce que k est directement à l'initialisation ?
        self.I = np.zeros((m,self.k))
        self.U = np.zeros((n,self.k))

    def train_GD(self, R, tau):
        """
        This function will train the model with parameter k using gradient descent (slide 22).
        """

        cost_function = np.trace()
        # TODO

    def train_ALS(self,R):
        """
        This function will train the model with parameter k using ALS (slide 22).
        """
        T = 1000
        lmbd = 1
        mu = 1
        for i in range(T):
            (self.I, self.U) = (R@ self.U @np.linalg.inv(self.U.T@self.U + lmbd *np.eye(self.k)),R.T@I@np.linalg.inv(self.I.T@self.I+ mu*np.eye(self.k)))

    def predict(self, R):
        """
        This function will be used to predict R_hat.
        """

    def score_RMSE(self, k, R):
        """
        This function will return the RMSE score.
        """
        R_hat = np.array()

        return metric.RMSE(R, R_hat)

    def score_0_1loss(self, R):
        """
        This function will return the 0-1 loss score.
        """
        R_hat = np.array()
        return metric.loss0_1(R, R_hat)


