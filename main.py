import numpy as np
import metric
import matrix_factorisation as MF
import time

# LOADING THE DATA
data_train = np.load("ratings_train.npy")   # shape = (610, 4980)
# Rating table: row = user ; column = rating

data_test = np.load("ratings_test.npy")     # shape = (610, 4980)
# Rating table: row = user ; column = rating

data_genre = np.load("namesngenre.npy")     # shape = (4980, 2)
# data_genre contains [movies name and genre]


# PREPROCESSING OF THE TRAINING DATA
data_train = np.nan_to_num(data_train, nan=np.nanmean(data_train, axis=1, keepdims=True))

#average = np.nanmean(data_train)
#data_train = np.nan_to_num(data_train, nan=average)

m,n = data_train.shape
method = MF.matrix_factorisation(k=1, m=m, n=n)

# TRAINING START
print("Start training")
start_time = time.time()
#method.train_GD(data_train, 0.001, 0.001, lmbda=0, mu=0, nb_ite=100, alternate_counter=20)
method.train_ALS(data_train,lmbda=1, mu=1)
end_time = time.time()
print("Training time =", end_time-start_time, "s")
#average = np.nanmean(data_test)

# TESTING ON TESTING DATA
R_isnan = np.isnan(data_test)       # R_isnan contains an array of shape R.shape, True if the element is NaN, False otherwise
T = np.argwhere(R_isnan == False)    # T contains a list of list containing the indices where R is NOT NaN
data_test = np.nan_to_num(data_test)

prediction = method.predict(round=True)
print("Score RMSE =", metric.RMSE(data_test, prediction, T))
print("Score accuracy =", metric.accuracy(data_test, prediction, T))
#TODO:
# Step 1: Modify the nan elements of the dataset
# Step 2: Use an implementation
# Step 3: Judge the performance

