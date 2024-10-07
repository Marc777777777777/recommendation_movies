import numpy as np
import metric
import matplotlib.pyplot as plt

class matrix_factorisation:
    """
    An object of this class will implement the training and 
    """
    def __init__(self, k, m, n, random_init=False, seed=0):
        """
        The object will have the I matrix and the U matrix.
        The training will use some methods to 
        """
        self.k = k
        if random_init:
            np.random.seed(seed)
            ratings = np.arange(0.5, 5.5, 0.5)
            self.I = np.random.choice(ratings, (m, self.k))
            self.U = np.random.choice(ratings, (n, self.k))
        else:
            self.I = np.ones((m, self.k))
            self.U = np.ones((n, self.k))


    def train_GD(self, R, tau_U=0.1, tau_I=1e-10, lmbda=0.001, mu=0.001, nb_ite=500, alternate_counter=25):
        """
        This function will train the model with parameter k using gradient descent (slide 22).
        """
        cost_function = cost(R, self.I, self.U, lmbda=lmbda, mu=mu)
        cost_val = [cost_function]
        counter = 0
        for ite in range(nb_ite):
            print("ite numero:",ite)

            if counter < alternate_counter:
                C_deriv_U = -2*R.T@self.I + 2*self.U@self.I.T@self.I + 2*mu*self.U
                self.U = self.U - tau_U*C_deriv_U
                counter += 1
                
                
            else:
                C_deriv_I = -2*R@self.U + 2*self.I@self.U.T@self.U + 2*lmbda*self.I
                self.I = self.I - tau_I*C_deriv_I
                counter += 1

                if counter >= 2*alternate_counter:
                    counter = 0
        
            cost_val.append(cost(R, self.I, self.U, lmbda=lmbda, mu=mu))
            print("Cost =", cost_val[-1])
            print("Max U =", np.max(self.U))
            print("Max Grad U =", np.max(C_deriv_U))
            #print("Max I =", np.max(self.I))
            #print("Max Grad I =", np.max(C_deriv_I))
            print()

        #plt.plot(np.arange(0, nb_ite+1, 1), cost_val)
        #plt.semilogy(np.arange(0, nb_ite+1, 1), cost_val)
        #plt.show()

    def train_ALS(self, R, lmbda=1., mu=1., nb_ite=200, alternate_counter=25):
        """
        This function will train the model with parameter k using ALS (slide 22).
        """
        cost_function = cost(R, self.I, self.U, lmbda=lmbda, mu=mu)
        #cost_val = [cost_function]
        counter = 0

        for ite in range(nb_ite):
            #(self.I, self.U) = (R@ self.U @np.linalg.inv(self.U.T@self.U + lmbda *np.eye(self.k)), R.T@self.I@np.linalg.inv(self.I.T@self.I+ mu*np.eye(self.k)))
            #print("Iteration:", ite)
            if counter < alternate_counter:
                self.U = R.T@self.I@np.linalg.inv(self.I.T@self.I+ mu*np.eye(self.k))
                counter += 1
            
            else:
                self.I = R@ self.U @np.linalg.inv(self.U.T@self.U + lmbda *np.eye(self.k))
                counter += 1

                if counter >= 2*alternate_counter:
                    counter = 0

            #cost_val.append(cost(R, self.I, self.U, lmbda=lmbda, mu=mu))

        #plt.plot(np.arange(0, nb_ite+1, 1), cost_val)
        #plt.semilogy(np.arange(0, nb_ite+1, 1), cost_val)
        #plt.show()
    
    def predict(self, round=True):
        """
        This function will be used to predict R_hat.
        """
        R_hat = self.I@self.U.T
        if round:
            R_hat = np.round(2*R_hat)/2
        return R_hat

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

def cost(R, I, U, lmbda=1., mu=1.):
    """
    This function returns the cost function defined in the slide.
    """
    return np.trace(R.T@R) - 2*np.trace(R.T@I@U.T) + np.trace(U@I.T@I@U.T) + lmbda*np.trace(U.T@U) + mu*np.trace(I.T@I)

def roundup_prediction(R_hat):
    """
    This function should take a rating matrix that is not exact (the rates are not among the possible one),
    and it will roundup the elements to the closest correct rate. 
    """
    R_hat_roundup = np.zeros(R_hat.shape)
    return R_hat_roundup