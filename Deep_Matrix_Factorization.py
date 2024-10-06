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
        return genres_padded
    
    def train_DMF(self, R, data_genre, genre_embedding_dim, latent_dim, epochs, learning_rate, num_layers):
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
        def create_dmf_model(num_users, num_items, num_genres, latent_dim):
            user_input = Input(shape=(num_items,), name='user_input')  # Prendre tout le vecteur Y_{i*} (toutes les notes d'un utilisateur)
            item_input = Input(shape=(num_users,), name='item_input')  # Prendre tout le vecteur Y_{*j} (toutes les notes d'un film)
 
            genre_input = Input(shape=(max_genre_length,), name='genre_input')  # Entrée flexible pour plusieurs genres par film
   
            # Embedding des genres
            genre_embedding = Embedding(input_dim=num_genres, output_dim=genre_embedding_dim)(genre_input)  
            # Appliquer une réduction (moyenne) sur les embeddings des genres
            genre_embedding_mean = Lambda(lambda x: tf.reduce_mean(x, axis=1))(genre_embedding)  # Moyenne des embeddings des genres

            # Ajouter des couches dynamiques
            user_projection = Dense(latent_dim, activation='relu')(user_input)
            item_projection = Dense(latent_dim, activation='relu')(item_input)
            
            # Ajouter plusieurs couches dynamiques en fonction du nombre spécifié
            for _ in range(num_layers - 1):  # num_layers - 1 car on a déjà une couche
                user_projection = Dense(latent_dim, activation='relu')(user_projection)
                item_projection = Dense(latent_dim, activation='relu')(item_projection)
            
            # Combiner l'embedding moyen des genres avec la projection de l'item
            item_with_genre = Concatenate()([item_projection, genre_embedding_mean])
            
            # Projeter item_with_genre sur la même dimension que user_projection
            item_with_genre = Dense(latent_dim, activation='relu')(item_with_genre)
            
            # Combiner les deux projections avec une similarité cosinus
            similarity = Dot(axes=1, normalize=True)([user_projection, item_with_genre])*5
    
            model = Model(inputs=[user_input, item_input, genre_input], outputs=similarity)
            return model
    
        # Étape 4 : Entraîner le modèle avec rétropropagation
        def train_dmf_model(model, Y, genres_padded, epochs, learning_rate):
            optimizer = Adam(learning_rate=learning_rate)
    
            user_vectors, item_vectors = self.prepare_data_for_training(Y)

            positives = np.argwhere(Y > 0)  # Trouver les notes observées

            users_pos = positives[:, 0]
            items_pos = positives[:, 1]
            rating_pos = Y[users_pos,items_pos]

            # Compilation du modèle
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            
            # Entraînement du modèle
            model.fit([user_vectors[users_pos], item_vectors[items_pos],
                       tf.keras.preprocessing.sequence.pad_sequences(genres_padded, padding='post')[items_pos]], rating_pos, epochs=epochs)

    
        # Préparation des données d'entraînement et de test
        Y_train = prepare_interaction_matrix(R)
        genres_padded, num_genres, max_genre_length = prepare_genres(data_genre)

        num_users, num_items = Y_train.shape
    
        # Création du modèle DMF
        model = create_dmf_model(num_users, num_items, num_genres, latent_dim)
    
        # Entraînement du modèle avec les données d'entraînement
        train_dmf_model(model, Y_train, genres_padded, epochs, learning_rate)
        
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
 
    
    def predict(self, model, user_vectors, item_vectors, genres_padded, batch_size):
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
                genre_batch = genres_padded[start_i:end_i]  # Extraire les genres des films pour ce batch
                
                # Créer toutes les combinaisons utilisateur-item dans ce batch
                user_repeat_batch = np.repeat(user_batch, item_batch.shape[0], axis=0)
                item_tile_batch = np.tile(item_batch, (user_batch.shape[0], 1))
                genre_tile_batch = np.tile(genre_batch, (user_batch.shape[0], 1))  # Étendre les genres pour correspondre aux combinaisons

                
                # Prédiction sur le batch
                R_hat_batch = model.predict([user_repeat_batch, item_tile_batch, genre_tile_batch], verbose=0)
                
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