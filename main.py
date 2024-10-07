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
R_isnan = np.isnan(data_train)        #
T_train = np.argwhere(R_isnan == False)
data_train = np.nan_to_num(data_train, nan=np.nanmean(data_train, axis=1, keepdims=True))
m,n = data_train.shape
#average = np.nanmean(data_train)
#data_train = np.nan_to_num(data_train, nan=average)

# TESTING ON TESTING DATA
R_isnan = np.isnan(data_test)        # R_isnan contains an array of shape R.shape, True if the element is NaN, False otherwise
T_test = np.argwhere(R_isnan == False)    # T contains a list of list containing the indices where R is NOT NaN

# List for plots:
k_values = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
#k_values = [1,1,1,1,1, 20,20,20,20,20, 40,40,40,40,40, 60,60,60,60,60, 80,80,80,80,80, 100,100,100,100,100, 120,120,120,120,120, 140,140,140,140,140, 160,160,160,160,160, 180,180,180,180,180, 200,200,200,200,200]
lmbda=1
mu=0.5
RMSE_values_train = []
accuracy_values_train = []
RMSE_values_test = []
accuracy_values_test = []
time_values = []

random_RMSE_values_train = []
random_accuracy_values_train = []
random_RMSE_values_test = []
random_accuracy_values_test = []

for k in k_values:
    # TRAINING START
    print("Start training, k =", k)
    method = MF.matrix_factorisation(k=k, m=m, n=n)
    #method_random = MF.matrix_factorisation(k=k, m=m, n=n, random_init=True,)
    start_time = time.time()
    #method.train_GD(data_train, 0.001, 0.001, lmbda=0, mu=0, nb_ite=100, alternate_counter=20)
    method.train_ALS(data_train, lmbda=lmbda, mu=mu)
    end_time = time.time()
    time_values.append(end_time-start_time)
    print("Training time =", time_values[-1], "s")
    
    random_RMSE = 0
    random_accuracy= 0
    for seed in range(5):
        random_method = MF.matrix_factorisation(k=k, m=m, n=n, random_init=True, seed=seed)
        random_method.train_ALS(data_train, lmbda=lmbda, mu=mu)
        prediction_random = random_method.predict(round=True)
        random_RMSE += metric.RMSE(data_test, prediction_random, T_test)
        random_accuracy += metric.accuracy(data_test, prediction_random, T_test)

    random_RMSE_values_test.append(random_RMSE/5)
    random_accuracy_values_test.append(random_accuracy/5)
    prediction = method.predict(round=True)
    #RMSE_values_train.append(metric.RMSE(data_train, prediction, T_train))
    #accuracy_values_train.append(metric.accuracy(data_train, prediction, T_train))
    RMSE_values_test.append(metric.RMSE(data_test, prediction, T_test))
    accuracy_values_test.append(metric.accuracy(data_test, prediction, T_test))
    print("Score RMSE test =", RMSE_values_test[-1])
    print("Score accuracy test =", accuracy_values_test[-1])


# PLOTTING THE PERFORMANCE
#plt.plot(k_values, RMSE_values_train, label="Training RMSE", color='b')
plt.plot(k_values, RMSE_values_test, label="Testing RMSE", color='r')
plt.plot(k_values, random_RMSE_values_test, label="Testing RMSE random init", color='g')
#plt.scatter(k_values, RMSE_values_train)
plt.title("RMSE score depending on k")
plt.xlabel("k")
plt.ylabel("RMSE")
plt.legend()
plt.grid(visible=True)
plt.show()

#plt.plot(k_values, accuracy_values_train, label="Training accuracy", color='b')
plt.plot(k_values, accuracy_values_test, label="Testing accuracy", color='r')
plt.plot(k_values, random_accuracy_values_test, label="Testing RMSE random init", color='g')
#plt.scatter(k_values, accuracy_values_test)
plt.title("Accuracy score depending on k")
plt.xlabel("k")
plt.ylabel("accuracy")
plt.legend()
plt.grid(visible=True)
plt.show()

#plt.plot(k_values, time_values)
##plt.scatter(k_values, time_values)
#plt.title("Training time depending on k")
#plt.xlabel("k")
#plt.ylabel("Training time")
#plt.legend()
#plt.show()

