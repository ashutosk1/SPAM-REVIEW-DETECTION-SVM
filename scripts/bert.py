import tensorflow as tf
from transformers import TFAutoModel
from transformers import AutoTokenizer


class BERTForClassification(tf.keras.Model):

    """
    Initializes the BERTForClassification model with hyperparameters.
    """

    def __init__(self, model_name, seq_length):
        super().__init__()
        self.model_name = model_name
        self.seq_length = seq_length

        self.bert = TFAutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Define the classification head
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(8, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')


    def call(self, inputs):

        """
        Forward pass of the model.
        Args:
            inputs (dict or list): Input dictionary for the model - "input_ids" and "attention_masks" as keys
        Returns:
            tf.Tensor: Model output logits
        """

        x = self.bert(inputs)[0][:, 0, :]
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.dropout3(x)
        outputs = self.output_layer(x)
        return outputs


    def tokenize(self, texts):
        """
        Encodes the text with pre-trained BERT-based tokenizer
        """
        encoded_texts = self.tokenizer( texts,
                                        max_length=self.seq_length,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='tf'
                                    )
        return [
                encoded_texts['input_ids'],
                encoded_texts['attention_mask']
                ]
    
    
    def train(self, train_feats, train_labels, val_feats, val_labels, epochs, batch_size):
        """
        Trains the model on the provided data.
        Steps:
        ** Encoding the reviews using pre-trained BERT-based tokenizer and accessing a list of tensors 
            as input_ids and attention_masks.**
        ** Summarize the model **
        ** Compile the model with appropriate loss_fn, optimizer and metrics**
        ** Fit, train and return the history object. **
        """

        train_feats_encoded = self.tokenize(train_feats.tolist())
        val_feats_encoded   = self.tokenize(val_feats.tolist())


        # Compile the Model
        self.compile(
                    loss='binary_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy']
                )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.00001)

        history = self.fit( x=train_feats_encoded, 
                            y=train_labels,
                            validation_data=(val_feats_encoded, val_labels),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping, reduce_lr]
                        )
        return history



