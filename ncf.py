import torch

import torch

class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()

        # Fetch the configuration values from the config dictionary
        self.num_users = config['num_users']  # Add this line to set the number of users
        self.num_items = config['num_items']  # Add this line to set the number of items
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']
        self.config = config  # Store the config for dropout later

        # MF part
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        # MLP part
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)

        # Fully connected layers in the MLP part
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # Final layer (logits)
        self.logits = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices, titles=None):
        # User and item embeddings for MLP and MF
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # MF part
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
        mf_vector = torch.nn.Dropout(self.config['dropout_rate_mf'])(mf_vector)

        # MLP part
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # Concatenate user & item embeddings
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)
        mlp_vector = torch.nn.Dropout(self.config['dropout_rate_mlp'])(mlp_vector)

        # Concatenate MF and MLP parts
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.logits(vector)
        output = self.sigmoid(logits).squeeze()

        return output

    def train_model(self, train_loader, loss_function, optimizer, num_epochs):
        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode

            total_loss = 0.0  # Variable to accumulate loss

            # Iterate through each batch of training data
            for user_indices, item_indices, labels in train_loader:
                optimizer.zero_grad()  # Clear previous gradients

                # Forward pass: Compute predictions
                outputs = self.forward(user_indices, item_indices, None)  # Assuming titles are not used

                # Compute loss: Compare predictions to actual labels
                loss = loss_function(outputs, labels.float())

                # Backward pass: Compute gradients
                loss.backward()  # Calculate gradients

                # Update weights: Adjust model parameters based on gradients
                optimizer.step()  # Perform optimization step

                total_loss += loss.item()  # Accumulate loss for this batch

            avg_loss = total_loss / len(train_loader)  # Average loss for the epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")