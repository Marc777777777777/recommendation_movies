import numpy as np
import torch
import os
from tqdm import tqdm, trange
import argparse
import matrix_factorisation as MF
import Deep_Matrix_Factorization as DMF
import ncf
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

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
    """
    table = np.nan_to_num(table)
    mf_model = DMF.matrix_factorisation()
    model = mf_model.train_DMF(table, latent_dim=64, epochs=10, learning_rate=0.001, layers=2)
    user_vectors, item_vectors = mf_model.prepare_data_for_training(table)
    table = mf_model.predict(model, user_vectors, item_vectors, batch_size=100)
    """

    ###NCF
    def rmse_loss(outputs, labels):
        mse_loss = torch.nn.functional.mse_loss(outputs, labels)
        return torch.sqrt(mse_loss)
    non_nan_indices = ~np.isnan(table)
    user_indices, item_indices = np.where(non_nan_indices)
    ratings = table[non_nan_indices]
    users = torch.tensor(user_indices, dtype=torch.long)
    items = torch.tensor(item_indices, dtype=torch.long)
    labels = torch.tensor(ratings, dtype=torch.float32)
    train_dataset = TensorDataset(users, items, labels)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    config = {
        'layers': [64,32, 16, 8],
        'latent_dim_mf': 8,
        'latent_dim_mlp': 32,
        'dropout_rate_mf': 0.2,
        'dropout_rate_mlp': 0.5,
        'num_users': 610,
        'num_items': 4980
    }
    method = ncf.NeuMF(config)
    num_epochs = 20
    learning_rate = 0.001
    optimizer = torch.optim.SGD(method.parameters(), lr=learning_rate)
    method.train_model(train_loader,rmse_loss, optimizer, num_epochs)

    nan_indices = np.argwhere(np.isnan(table))
    nan_users = nan_indices[:, 0]
    nan_items = nan_indices[:, 1]
    prediction = method.forward(torch.tensor(nan_users, dtype=torch.long), torch.tensor(nan_items, dtype=torch.long))

    for i, pred in zip(nan_indices, prediction):
        table[i[0], i[1]] = pred.item()


    # Save the completed table
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE
