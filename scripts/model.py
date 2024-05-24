from sklearn.pipeline import Pipeline
from nltk.classify import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# lstm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D


import pandas as pd
import numpy as np
from random import shuffle

import warnings
warnings.filterwarnings("ignore")

# local
from preprocess import lstm_preprocess



def trainer(processed_review_df, config):
    """
    Trains and evaluates the 
    classification model using cross-validation. 
    Steps:
    ** Extract Feature Vectors and Labels from preprocessed review dataset.**
    ** Build Classification Model with `model_name` and `max_iter` as the params.**
    ** Get Cross-Validation score by fitting the model on the train set and evaluating on validation set.** 
    """

    if config["MODEL_NAME"] =="LSTM":
        # Preprocess for the LSTM Input Layer : Tokenize, `fit.fit_on_texts()`, `texts_to_sequences()`, `pad_sequences`.
        processed_review_df, _ = lstm_preprocess(processed_review_df, config["LSTM"]["num_words"], config["LSTM"]["max_length"])
    
    processed_review_df_feats = np.array(processed_review_df["FEATURE_VECTOR"])
    processed_review_df_labels= processed_review_df["LABEL"].to_numpy()
    
    cv_scores = get_cross_validation_acc(processed_review_df_feats, processed_review_df_labels, config)  

    cv_scores_dict =    {
      "accuracy"  : round(cv_scores[0], 3),
      "precision" : round(cv_scores[1], 3),
      "recall"    : round(cv_scores[2], 3),
      "f-score"   : round(cv_scores[3], 3)
    }
    return cv_scores_dict                 



def get_sklearn_model_pipeline(config):
    """
    Constructs and returns a classification model pipeline to br trained with `SklearnClassifier`.
    """
    # Define the model based on model_name
    if config["MODEL_NAME"]== "LR":
        model = LogisticRegression(max_iter=config["common"]["max_iter"])
        pipeline =  Pipeline([('clf', model)])
    elif config["MODEL_NAME"]== "SVM":
        model = LinearSVC()
        pipeline =  Pipeline([('svc', model)])
    else:
        raise ValueError(f"Invalid model name")
    return pipeline


def get_lstm_model(max_words, max_len, embedding_dim, lstm_units):
    """
    Builds an LSTM model for text classification.
    """   
    # build layers
    model =  Sequential()
    model.add(Embedding(max_words,embedding_dim, input_length=max_len))
    model.add(SpatialDropout1D(0.5))
    model.add(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))

    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_cross_validation_acc(feats, labels, config):
    """
    Performs k-fold cross-validation and prints performance metrics for each fold.
    """

    assert len(feats) == len(labels)

    fold_size = len(labels)// (config["FOLD"])
    cv_scores = []

    # Shuffle
    merged_feats_and_labels = list(zip(feats, labels))
    shuffle(merged_feats_and_labels)
    feats, labels = zip(*merged_feats_and_labels)

    for i in range(config["FOLD"]):
        # Validation set
        val_feats = np.array(feats[i*fold_size:(i+1)*fold_size])
        val_labels = np.array(labels[i*fold_size:(i+1)*fold_size])

        # Training set
        train_feats = np.array(feats[:i*fold_size] + feats[(i+1)*fold_size:])
        train_labels = np.array(labels[:i*fold_size] + labels[(i+1)*fold_size:])

        if config["MODEL_NAME"] in ["LR", "SVM"]:
            pipeline = get_sklearn_model_pipeline(config)
            print(f"\n\t*** TRAINING AND CROSS VALIDATION: {config['MODEL_NAME']} ***\t")
            classifier = SklearnClassifier(pipeline).train(list(zip(train_feats, train_labels)))
            val_preds = classifier.classify_many(val_feats)
        
        if config["MODEL_NAME"] == "LSTM":
            model = get_lstm_model(config["LSTM"]["num_words"], config["LSTM"]["max_length"], config["LSTM"]["embed_dim"], config["LSTM"]["lstm_units"])
            print(f"\n\t*** TRAINING AND CROSS VALIDATION: {config['MODEL_NAME']} ***\t")
            history = model.fit(train_feats, train_labels, epochs = config["LSTM"]["epoch"], batch_size = config["LSTM"]["batch_size"])
            val_preds = model(val_feats).numpy().flatten()
            val_preds = (val_preds >= 0.5).astype(int)


        acc = accuracy_score(val_labels, val_preds)
        (p, r, f, _) = precision_recall_fscore_support(y_pred=val_preds, y_true=val_labels, average='macro')

        print(f"\nfold:{i+1}, acc:{acc:.3f}, precision:{p:.3f}, recall:{r:.3f}, f-score:{f:.3f}")
        cv_scores.append((acc, p, r, f))

    cv_scores = np.mean(np.array(cv_scores), axis=0)
    return cv_scores
