import numpy as np

def RMSE(R, R_hat, T):
    """
    This function compute the RMSE (slide 30).
    """
    difference_squared = 0
    for i in range(len(T)):
        difference_squared += ( R[T[i][0], T[i][1]] - R_hat[T[i][0], T[i][1]])**2

    return np.sqrt( difference_squared / len(T) )

def accuracy(R, R_hat, T):
    """
    Accuracy of the slides
    """
    accu = 0
    for i in range(len(T)):
        if R[T[i][0], T[i][1]] == R_hat[T[i][0], T[i][1]]:
            accu += 1 # If the rating in R and R_hat are different, then we add 1
        
    return accu   # if difference is small (near 0), then the ratings are the same for a lot of elements

def loss0_1(R, R_hat, T):
    """
    This function compute the 0-1 loss for element that are in R.
    """
    difference = 0
    for i in range(len(T)):
        if R[T[i][0], T[i][1]] != R_hat[T[i][0], T[i][1]]:
            difference += 1 # If the rating in R and R_hat are different, then we add 1
        
    return difference   # if difference is small (near 0), then the ratings are the same for a lot of elements

