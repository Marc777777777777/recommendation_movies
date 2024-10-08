import numpy as np
import os
from tqdm import tqdm, trange
import argparse
import matrix_factorisation as MF
import Deep_Matrix_Factorization as DMF


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    data_genre = np.load("namesngenre.npy")  
    print('Ratings Loaded.')
    

    # Any method you want
    '''
    ### MATRIX FACTORISATION ###
    #average = np.nanmean(table)
    #table = np.nan_to_num(table, nan=average)
    table = np.nan_to_num(table, nan=np.nanmean(table, axis=1, keepdims=True))
    m,n = table.shape
    # method = MF.matrix_factorisation(k=4, m=m, n=n)
    # method.train_ALS(table, lmbda=1, mu=1)
    method = MF.matrix_factorisation(k=90, m=m, n=n)
    method.train_ALS(table, lmbda=0.1, mu=0.1)
    table = method.predict()
    '''
    ### DEEP MATRIX FACTORISATION ###
    table = np.nan_to_num(table)
    mf_model = DMF.matrix_factorisation()
    model = mf_model.train_DMF(table, data_genre, genre_embedding_dim=16, latent_dim=124, epochs=30, learning_rate=0.001, num_layers=2)
    user_vectors, item_vectors = mf_model.prepare_data_for_training(table)
    item_genre = mf_model.prepare_genres(data_genre)
    table = mf_model.predict(model, user_vectors, item_vectors, item_genre, batch_size=100)

    




    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE
