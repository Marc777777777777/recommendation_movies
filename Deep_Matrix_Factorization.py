import numpy as np
import metric
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Multiply, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dot
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class matrix_factorisation:
    """
    An object of this class will implement the training and 
    """

        
    def prepare_data_for_training(self, Y):
        num_users, num_items = Y.shape
        user_vectors = np.zeros((num_users, num_items))  # Vecteurs utilisateurs Y_{i*}
        item_vectors = np.zeros((num_items, num_users))  # Vecteurs items Y_{*j}
        
        # Remplir les vecteurs
        for i in range(num_users):
            user_vectors[i, :] = Y[i, :]  # Vecteur des notes pour l'utilisateur i
        
        for j in range(num_items):
            item_vectors[j, :] = Y[:, j]  # Vecteur des notes pour l'item j
        
        return user_vectors, item_vectors
    
    def train_DMF(self, R, latent_dim, epochs, learning_rate, num_layers):
        '''
        Fonction unique pour Deep Matrix Factorization, prenant en compte une matrice R avec des NaN pour les notes manquantes.
        
        R : Matrice utilisateur-film avec des notes et des NaN (notes manquantes)
        latent_dim : Dimension des embeddings
        neg_ratio : Ratio d'échantillons négatifs par rapport aux positifs
        epochs : Nombre d'époques d'entraînement
        batch_size : Taille du batch pour l'entraînement
        learning_rate : Taux d'apprentissage
        '''
        # Étape 1 : Préparer la matrice d'interaction Y
        def prepare_interaction_matrix(R):
            Y = np.nan_to_num(R, nan=0.0)  # Remplacer les NaN par 0
          
            return Y   
        
        
        # Étape 2 : Générer des échantillons négatifs
        def generate_negative_samples(Y, neg_ratio):
            positives = np.argwhere(Y > 0)  # Interactions observées
            negatives = []
            num_positives = len(positives)
            for _ in range(num_positives * neg_ratio):
                user = np.random.randint(0, Y.shape[0])
                item = np.random.randint(0, Y.shape[1])
                if Y[user, item] == 0:  # Ajouter seulement si non observé
                    negatives.append([user, item])
            return positives, np.array(negatives)
        
        # Étape 3 : Créer le modèle DMF
        def create_dmf_model(num_users, num_items, latent_dim):
            user_input = Input(shape=(num_items,), name='user_input')  # Prendre tout le vecteur Y_{i*} (toutes les notes d'un utilisateur)
            item_input = Input(shape=(num_users,), name='item_input')  # Prendre tout le vecteur Y_{*j} (toutes les notes d'un film)
 
            # Ajouter des couches dynamiques
            user_projection = Dense(latent_dim, activation='relu')(user_input)
            item_projection = Dense(latent_dim, activation='relu')(item_input)
            
            # Ajouter plusieurs couches dynamiques en fonction du nombre spécifié
            for _ in range(num_layers - 1):  # num_layers - 1 car on a déjà une couche
                user_projection = Dense(latent_dim, activation='relu')(user_projection)
                item_projection = Dense(latent_dim, activation='relu')(item_projection)
            
            # Combiner les deux projections avec une similarité cosinus
            similarity = Dot(axes=1, normalize=True)([user_projection, item_projection])*5
    
            model = Model(inputs=[user_input, item_input], outputs=similarity)
            return model
    
        # Étape 4 : Entraîner le modèle avec rétropropagation
        def train_dmf_model(model, Y, epochs, learning_rate):
            optimizer = Adam(learning_rate=learning_rate)
    
            user_vectors, item_vectors = self.prepare_data_for_training(Y)

            positives = np.argwhere(Y > 0)  # Trouver les notes observées

            users_pos = positives[:, 0]
            items_pos = positives[:, 1]
            rating_pos = Y[users_pos,items_pos]

    
            # Compilation du modèle
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            
            # Entraînement du modèle
            model.fit([user_vectors[users_pos], item_vectors[items_pos]], rating_pos, epochs=epochs)

    
        # Préparation des données d'entraînement et de test
        Y_train = prepare_interaction_matrix(R)
    
        num_users, num_items = Y_train.shape
    
        # Création du modèle DMF
        model = create_dmf_model(num_users, num_items, latent_dim)
    
        # Entraînement du modèle avec les données d'entraînement
        train_dmf_model(model, Y_train, epochs, learning_rate)
        
        return model
    def create_dynamic_dmf_model(self, num_users, num_items, latent_dim, num_layers):
        user_input = Input(shape=(num_items,), name='user_input')
        item_input = Input(shape=(num_users,), name='item_input')
    
        # Ajouter des couches dynamiques
        user_projection = Dense(latent_dim, activation='relu')(user_input)
        item_projection = Dense(latent_dim, activation='relu')(item_input)
        
        # Ajouter plusieurs couches dynamiques en fonction du nombre spécifié
        for _ in range(num_layers - 1):  # num_layers - 1 car on a déjà une couche
            user_projection = Dense(latent_dim, activation='relu')(user_projection)
            item_projection = Dense(latent_dim, activation='relu')(item_projection)
    
        # Calculer la similarité cosinus entre les deux projections
        similarity = Dot(axes=1, normalize=True)([user_projection, item_projection]) * 5
    
        model = Model(inputs=[user_input, item_input], outputs=similarity)
        return model   
    
    def train_and_optimize(self, model, Y, user_vectors, item_vectors, params):
        optimizer = Adam(learning_rate=params['learning_rate'])
    
        positives = np.argwhere(Y > 0)  # Trouver les notes observées

        users_pos = positives[:, 0]
        items_pos = positives[:, 1]
        rating_pos = Y[users_pos,items_pos]
        
        user_vectors_observed = user_vectors[users_pos]
        item_vectors_observed = item_vectors[items_pos]

        # Compilation du modèle
        model.compile(optimizer=optimizer, loss='mean_squared_error')
            
        # Définir les callbacks pour arrêter l'entraînement ou ajuster le learning rate si nécessaire
        early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.0001)
        
        # Entraîner le modèle et surveiller l'erreur (RMSE) à chaque époque
        history = model.fit([user_vectors_observed, item_vectors_observed], rating_pos,
                            epochs=params['epochs'], batch_size=32, verbose=1,
                            callbacks=[early_stopping, reduce_lr])
    
        # Récupérer l'historique de perte pendant l'entraînement
        loss_history = history.history['loss']
        
        return loss_history[-1]  # Retourne la perte à la dernière époque
 
    
    def predict(self, model, user_vectors, item_vectors, batch_size):
        # Créer des matrices d'indices pour chaque utilisateur et chaque item
        num_users = user_vectors.shape[0]
        num_items = item_vectors.shape[0]
            
        R_hat = np.zeros((num_users, num_items))  # Matrice pour stocker les prédictions
        
        # Diviser les prédictions en sous-batches
        #Cette boucle permet de traiter les utilisateurs par petits groupes, 
        #de start_u à end_u, où batch_size fixe le nombre maximal d'utilisateurs 
        #à traiter à chaque itération.
        for start_u in range(0, num_users, batch_size):
            end_u = min(start_u + batch_size, num_users)
            user_batch = user_vectors[start_u:end_u]
            #Cette boucle imbriquée permet de traiter les items 
            #par petits groupes, de manière similaire à la boucle des utilisateurs :
            for start_i in range(0, num_items, batch_size):
                end_i = min(start_i + batch_size, num_items)
                item_batch = item_vectors[start_i:end_i]
                
                # Créer toutes les combinaisons utilisateur-item dans ce batch
                user_repeat_batch = np.repeat(user_batch, item_batch.shape[0], axis=0)
                item_tile_batch = np.tile(item_batch, (user_batch.shape[0], 1))
                
                # Prédiction sur le batch
                R_hat_batch = model.predict([user_repeat_batch, item_tile_batch], verbose=0)
                
                # Reshape et stocker les prédictions dans la matrice
                R_hat[start_u:end_u, start_i:end_i] = R_hat_batch.reshape((end_u - start_u, end_i - start_i))
        R_hat = np.around(2*R_hat)/2
        return R_hat

    def score_RMSE(self, k, R):
        R_hat = np.array()

        return metric.RMSE(R, R_hat)

    def score_0_1loss(self, R):
        R_hat = np.array()
        return metric.loss0_1(R, R_hat)