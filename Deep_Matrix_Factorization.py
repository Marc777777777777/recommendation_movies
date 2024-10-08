import numpy as np
import metric
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Multiply, ReLU, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dot
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    
    def prepare_genres(self, data_genre):
        # Séparer les genres multiples pour chaque film
        genres_split = [genre_str.split('|') for genre_str in data_genre[:, 1]]
            
        # Utiliser un Tokenizer pour encoder les genres
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(genres_split)
        genres_encoded = tokenizer.texts_to_sequences(genres_split)  # Encode les genres
        max_genre_length = max(len(genres) for genres in genres_encoded)  # Longueur maximale de genres
        genres_padded = pad_sequences(genres_encoded, maxlen=max_genre_length, padding='post')
        num_genres = len(tokenizer.word_index) + 1  # Nombre total de genres
        return genres_padded, num_genres, max_genre_length
    
    def train_DMF(self, R, data_genre, genre_embedding_dim, latent_dim, epochs, learning_rate, num_layers):
        '''
        Fonction unique pour Deep Matrix Factorization, prenant en compte une matrice R avec des NaN pour les notes manquantes.
        
        R : Matrice utilisateur-film avec des notes et des NaN (notes manquantes)
        latent_dim : Dimension des embeddings
        epochs : Nombre d'époques d'entraînement
        learning_rate : Taux d'apprentissage
        '''
        
        # Étape 1 : Préparer la matrice d'interaction Y
        def prepare_interaction_matrix(R):
            Y = np.nan_to_num(R, nan=0.0)  # Remplacer les NaN par 0
            return Y   
        
        # Étape 2 : Préparer les genres des films
        def prepare_genres(data_genre):
            # Séparer les genres multiples pour chaque film
            genres_split = [genre_str.split('|') for genre_str in data_genre[:, 1]]
            
            # Utiliser un Tokenizer pour encoder les genres
            tokenizer = tf.keras.preprocessing.text.Tokenizer()
            tokenizer.fit_on_texts(genres_split)
            genres_encoded = tokenizer.texts_to_sequences(genres_split)  # Encode les genres
            max_genre_length = max(len(genres) for genres in genres_encoded)  # Longueur maximale de genres
            genres_padded = pad_sequences(genres_encoded, maxlen=max_genre_length, padding='post')
            num_genres = len(tokenizer.word_index) + 1  # Nombre total de genres
            return genres_padded, num_genres, max_genre_length
        
        # Étape 3 : Créer le modèle DMF
        def create_dmf_model(num_users, num_items, num_genres, latent_dim, genre_embedding_dim):
            user_input = Input(shape=(num_items,), name='user_input')  # Prendre tout le vecteur Y_{i*} (notes utilisateur)
            item_input = Input(shape=(num_users + genre_embedding_dim,), name='item_input')  # Notes + genres
            
            # Projections utilisateur et item
            user_projection = Dense(latent_dim, activation='relu')(user_input)
            item_projection = Dense(latent_dim, activation='relu')(item_input)
            
            # Ajouter plusieurs couches dynamiques en fonction du nombre spécifié
            for _ in range(num_layers - 1):  # num_layers - 1 car on a déjà une couche
                user_projection = Dense(latent_dim, activation='relu')(user_projection)
                item_projection = Dense(latent_dim, activation='relu')(item_projection)
            
            # Combiner les deux projections avec une similarité cosinus
            similarity = Dot(axes=1, normalize=True)([user_projection, item_projection]) * 5
            
            model = Model(inputs=[user_input, item_input], outputs=similarity)
            return model
    
        # Étape 4 : Entraîner le modèle avec rétropropagation
        def train_dmf_model(model, Y, item_vectors_with_genre, epochs, learning_rate):
            optimizer = Adam(learning_rate=learning_rate)
    
            # Préparation des vecteurs utilisateur-item
            user_vectors, item_vectors = self.prepare_data_for_training(Y)
    
            positives = np.argwhere(Y > 0)  # Trouver les notes observées
    
            users_pos = positives[:, 0]
            items_pos = positives[:, 1]
            rating_pos = Y[users_pos, items_pos]
    
            # Compilation du modèle
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            
            # Entraînement du modèle
            model.fit([user_vectors[users_pos], item_vectors_with_genre[items_pos]], rating_pos, epochs=epochs)
    
        # Préparation des données d'entraînement et de test
        Y_train = prepare_interaction_matrix(R)
        genres_padded, num_genres, max_genre_length = prepare_genres(data_genre)
        user_vectors, item_vectors = self.prepare_data_for_training(Y_train)

        # Préparer les vecteurs items avec les genres
        item_vectors_with_genre = []
        for i in range(len(item_vectors)):
            combined_vector = np.concatenate((item_vectors[i], genres_padded[i]))  # Combiner les notes et les genres
            item_vectors_with_genre.append(combined_vector)
        # Convertir la liste en tableau NumPy
        item_vectors_with_genre = np.array(item_vectors_with_genre)

        num_users, num_items = Y_train.shape
    
        # Création du modèle DMF
        model = create_dmf_model(num_users, num_items, num_genres, latent_dim, genre_embedding_dim)
    
        # Entraînement du modèle avec les données d'entraînement
        train_dmf_model(model, Y_train, item_vectors_with_genre, epochs, learning_rate)
        
        return model
    
    def create_dynamic_dmf_model(self, num_users, num_items, genre_embedding_dim, latent_dim, num_layers):
        # Entrée utilisateur : vecteur de notes utilisateur
        user_input = Input(shape=(num_items,), name='user_input')
    
        # Entrée item : vecteur de notes item concaténé avec le genre
        item_input = Input(shape=(num_users + genre_embedding_dim,), name='item_input')
    
        # Projections utilisateur et item
        user_projection = Dense(latent_dim, activation='relu')(user_input)
        item_projection = Dense(latent_dim, activation='relu')(item_input)
    
        # Ajouter plusieurs couches dynamiques en fonction du nombre spécifié
        for _ in range(num_layers - 1):  # num_layers - 1 car on a déjà une couche
            user_projection = Dense(latent_dim, activation='relu')(user_projection)
            item_projection = Dense(latent_dim, activation='relu')(item_projection)
    
        # Combiner les deux projections avec une similarité cosinus
        similarity = Dot(axes=1, normalize=True)([user_projection, item_projection]) * 5
    
        # Créer le modèle
        model = Model(inputs=[user_input, item_input], outputs=similarity)
        return model
    
    def train_and_optimize(self, model, Y, user_vectors, item_vectors_with_genre, Y_test, 
                           user_vectors_test, item_vectors_with_genre_test, T, params, batch_size=100):
        
        optimizer = Adam(learning_rate=params['learning_rate'])
    
        # Trouver les interactions observées dans les données d'entraînement
        positives = np.argwhere(Y > 0)
        users_pos = positives[:, 0]
        items_pos = positives[:, 1]
        rating_pos = Y[users_pos, items_pos]
    
        # Préparer les vecteurs observés d'utilisateurs et d'items avec genres
        user_vectors_observed = user_vectors[users_pos]
        item_vectors_with_genre_observed = item_vectors_with_genre[items_pos]
    
        # Compilation du modèle
        model.compile(optimizer=optimizer, loss='mean_squared_error')
    
        # Définir les callbacks pour l'entraînement
        early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.0001)
    
        # Entraîner le modèle
        history = model.fit([user_vectors_observed, item_vectors_with_genre_observed], rating_pos,
                            epochs=params['epochs'], batch_size=batch_size, verbose=1,
                            callbacks=[early_stopping, reduce_lr])
    
        # Récupérer l'historique de perte pendant l'entraînement
        loss_history = history.history['loss']
        train_rmse = loss_history[-1]
    
        # Prédiction sur les données de test
        positives_test = np.argwhere(Y_test > 0)
        users_pos_test = positives_test[:, 0]
        items_pos_test = positives_test[:, 1]
    
        # Préparer les vecteurs observés pour le test
        user_vectors_test_observed = user_vectors_test[users_pos_test]
        item_vectors_with_genre_test_observed = item_vectors_with_genre_test[items_pos_test]
    
        # Créer une matrice pour stocker les prédictions
        R_hat = np.zeros((user_vectors_test.shape[0], item_vectors_with_genre_test.shape[0]))
    
        # Prédiction par batch
        for start_idx in range(0, len(users_pos_test), batch_size):
            end_idx = min(start_idx + batch_size, len(users_pos_test))
    
            user_batch = user_vectors_test_observed[start_idx:end_idx]
            item_batch = item_vectors_with_genre_test_observed[start_idx:end_idx]
    
            # Prédiction
            R_hat_batch = model.predict([user_batch, item_batch], verbose=0)
    
            # Stocker les prédictions
            for i, (u, it) in enumerate(zip(users_pos_test[start_idx:end_idx], items_pos_test[start_idx:end_idx])):
                R_hat[u, it] = R_hat_batch[i]
    
        R_hat = np.around(2 * R_hat) / 2  # Arrondir les prédictions à 0.5 près
    
        # Calculer la RMSE sur les données de test
        test_rmse = metric.RMSE(Y_test, R_hat, T)
    
        return train_rmse, test_rmse
    
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