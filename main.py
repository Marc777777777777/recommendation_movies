import numpy as np
import metric
import matrix_factorisation as MF
import time
import matplotlib.pyplot as plt

# LOADING THE DATA
data_train = np.load("ratings_train.npy")   # shape = (610, 4980)
# Rating table: row = user ; column = rating

data_test = np.load("ratings_test.npy")     # shape = (610, 4980)
# Rating table: row = user ; column = rating

data_genre = np.load("namesngenre.npy")     # shape = (4980, 2)
# data_genre contains [movies name and genre]


# PREPROCESSING OF THE TRAINING DATA
data_train = np.nan_to_num(data_train, nan=np.nanmean(data_train, axis=1, keepdims=True))
m,n = data_train.shape
#average = np.nanmean(data_train)
#data_train = np.nan_to_num(data_train, nan=average)

# TESTING ON TESTING DATA
R_isnan = np.isnan(data_test)        # R_isnan contains an array of shape R.shape, True if the element is NaN, False otherwise
T = np.argwhere(R_isnan == False)    # T contains a list of list containing the indices where R is NOT NaN

# List for plots:
#k_values = [1, 10, 50, 100, 200]
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
RMSE_values = []
accuracy_values = []
time_values = []

for k in k_values:

    # TRAINING START
    print("Start training, k =", k)
    method = MF.matrix_factorisation(k=k, m=m, n=n)
    start_time = time.time()

    #method.train_GD(data_train, 0.001, 0.001, lmbda=0, mu=0, nb_ite=100, alternate_counter=20)
    method.train_ALS(data_train, lmbda=1, mu=1)
    
    end_time = time.time()
    time_values.append(end_time-start_time)
    print("Training time =", time_values[-1], "s")

    prediction = method.predict(round=True)
    RMSE_values.append(metric.RMSE(data_test, prediction, T))
    accuracy_values.append(metric.accuracy(data_test, prediction, T))
    print("Score RMSE =", RMSE_values[-1])
    print("Score accuracy =", accuracy_values[-1])

# print(np.max(prediction))
# print(np.min(prediction))
# print(np.mean(prediction))
# PLOTTING THE PERFORMANCE
plt.plot(k_values, RMSE_values)
plt.title("RMSE score depending on k")
plt.xlabel("k")
plt.ylabel("RMSE")
plt.show()

plt.plot(k_values, accuracy_values)
plt.title("Accuracy score depending on k")
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()

plt.plot(k_values, time_values)
plt.title("Training time depending on k")
plt.xlabel("k")
plt.ylabel("Training time")
plt.show()

