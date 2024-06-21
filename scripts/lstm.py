import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class LSTMClassifier:
    """
    Build and train an LSTM model for text classification.
    """

    def __init__(self, max_words, max_len, embedding_dim, lstm_units):
        """
        Initialize the class with hyperparameters for the LSTM model.
        """
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None  # Initialize model as None


    def build_model(self):
        """
        Builds and returns the LSTM model.
        """
        if self.model is None:
            model = Sequential()
            model.add(Embedding(self.max_words, self.embedding_dim, input_length=self.max_len))
            model.add(LSTM(self.lstm_units, dropout=0.5, recurrent_dropout=0.2))
            model.add(Dropout(0.4))
            model.add(Dense(128, activation="relu"))
            model.add(Dropout(0.3))
            model.add(Dense(64, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(8, activation="relu"))
            model.add(Dropout(0.1))
            model.add(Dense(1, activation="sigmoid"))
            self.model = model

            # Compile
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model = model

        return self.model


    def train(self, train_feats, train_labels, val_feats, val_labels, epochs, batch_size):
        """
        Trains the LSTM model.
        """
        model = self.build_model()  

        validation_data = (val_feats, val_labels)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
        callbacks = [early_stopping, reduce_lr]

        history = model.fit(train_feats, train_labels, epochs=epochs, batch_size=batch_size,
                            validation_data=validation_data, callbacks=callbacks)

        return model, history

