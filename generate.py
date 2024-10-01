
import numpy as np
import os
from tqdm import tqdm, trange
import argparse
import matrix_factorisation as MF

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    

    # Any method you want
    average = np.nanmean(table)
    #table = np.nan_to_num(table, nan=average)
    table = np.nan_to_num(table, nan=np.nanmean(table, axis=1, keepdims=True))
    m,n = table.shape

    method = MF.matrix_factorisation(k=4, m=m, n=n)
    method.train_ALS(table, lmbda=1, mu=1)

    table = method.predict()

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
