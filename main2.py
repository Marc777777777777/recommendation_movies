import torch
import numpy as np
import metric
import ncf
import time
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def useritem(M):
    m,n = M.shape
    data = []
    for i in range(m):
        for j in range(n):
            label = (M[i,j])
            if not np.isnan(label):
                data.append((i,j,label))
    return data
# LOADING THE DATA
data_train = np.load("ratings_train.npy")   # shape = (610, 4980)
# Rating table: row = user ; column = rating

data_test = np.load("ratings_test.npy")     # shape = (610, 4980)
# Rating table: row = user ; column = rating

data_genre = np.load("namesngenre.npy")     # shape = (4980, 2)
# data_genre contains [movies name and genre]

#A modifier (tout mettre dans un tenseur directement)
data_train_list = useritem(data_train)

# Convert the data into separate tensors
users = torch.tensor([x[0] for x in data_train_list], dtype=torch.long)
items = torch.tensor([x[1] for x in data_train_list], dtype=torch.long)
labels = torch.tensor([x[2] for x in data_train_list], dtype=torch.float32)

# Create a TensorDataset
train_dataset = TensorDataset(users, items, labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


m,n = data_train.shape
config = {
    'layers': [32, 16, 8],      # MLP architecture - changed first layer from 64 to 32
    'latent_dim_mf': 8,         # MF embedding size
    'latent_dim_mlp': 16,       # MLP embedding size
    'dropout_rate_mf': 0.2,     # Dropout rate for MF part
    'dropout_rate_mlp': 0.2,    # Dropout rate for MLP part
    'num_users': 610,           # Number of users in the dataset
    'num_items': 4980           # Number of items in the dataset
}

method = ncf.NeuMF(config)

# TRAINING START
print("Start training")
start_time = time.time()

#training
num_epochs = 100
learning_rate = 0.001
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(method.parameters(), lr=learning_rate)
method.train_model(train_loader,loss_function, optimizer, num_epochs)



end_time = time.time()
print("Training time =", end_time-start_time, "s")