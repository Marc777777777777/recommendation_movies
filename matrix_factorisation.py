import numpy as np
import metric

class matrix_factorisation:
    """
    An object of this class will implement the training and 
    """
    def __init__(self, k, m, n):
        """
        The object will have the I matrix and the U matrix.
        The training will use some methods to 
        """
        ratings = np.arange(0.5, 5.5, 0.5)
        self.k = k
        self.I = np.random.choice(ratings, (m, self.k))
        self.U = np.random.choice(ratings, (n, self.k))
    
    def train_GD(self, R, tau1, tau2, lamba=0.5, mu=0.5, nb_ite=200):
        """
        This function will train the model with parameter k using gradient descent (slide 22).
        """

        for _ in range(nb_ite):
            C_deriv_U = -2*R.T@self.I + 2*self.U@self.I.T@self.I + 2*mu*self.U
            C_deriv_I = -2*R@self.U + 2*self.I@self.U.T@self.U + 2*lamba*self.I

            self.I = self.I - tau1*C_deriv_I
            self.U = self.U - tau2*C_deriv_U
        
    def train_ALS(self, R, lmbd=1, mu=1, nb_ite = 1000):
        """
        This function will train the model with parameter k using ALS (slide 22).
        """
        for _ in range(nb_ite):
            (self.I, self.U) = (R@ self.U @np.linalg.inv(self.U.T@self.U + lmbd *np.eye(self.k)), R.T@self.I@np.linalg.inv(self.I.T@self.I+ mu*np.eye(self.k)))

    
    def predict(self, R):
        """
        This function will be used to predict R_hat.
        """
        return self.I@self.U.T

    def score_RMSE(self, R, T):
        """
        This function will return the RMSE score.
        """
        R_hat = self.I@self.U.T
        return metric.RMSE(R, R_hat, T)

    def score_0_1loss(self, R, T):
        """
        This function will return the 0-1 loss score.
        """
        R_hat = self.I@self.U.T
        return metric.loss0_1(R, R_hat, T)
    
    def score_accuracy(self, R, T):
        """
        This function will return the accuracy of the slide
        """
        R_hat = self.I@self.U.T
        return metric.accuracy(R, R_hat, T)

