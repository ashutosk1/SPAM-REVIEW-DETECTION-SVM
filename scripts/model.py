from sklearn.metrics import accuracy_score, precision_recall_fscore_support



import pandas as pd
import numpy as np
from random import shuffle

import warnings
warnings.filterwarnings("ignore")

# local
from preprocess import lstm_preprocess, bert_preprocess
from sklearn_models import SklearnModelPipelineClassifier
from lstm import LSTMClassifier
from bert import BERTForClassification


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
        processed_review_df = lstm_preprocess(processed_review_df, config["LSTM"]["num_words"], config["LSTM"]["max_length"], config["FEATURES_LIST"])
    

    if config["MODEL_NAME"] == "BERT":
        # Preprocess for BERT Input Layer : Tokenization 
        print("IN THE BERT TOKENIZER")
        processed_review_df = bert_preprocess(processed_review_df, config["BERT"]["seq_length"], config["FEATURES_LIST"])
    

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



# def get_lstm_model(max_words, max_len, embedding_dim, lstm_units):
#     """
#     Builds an LSTM model for text classification.
#     """   
#     # build layers
#     model =  Sequential()
#     model.add(Embedding(max_words,embedding_dim, input_length=max_len))
#     model.add(LSTM(lstm_units, dropout=0.5, recurrent_dropout=0.2))
#     model.add(Dropout(0.4))
#     model.add(Dense(128, activation="relu"))
#     model.add(Dropout(0.3))
#     model.add(Dense(64, activation="relu"))
#     model.add(Dropout(0.2))
#     model.add(Dense(8, activation="relu"))
#     model.add(Dropout(0.1))
#     model.add(Dense(1, activation="sigmoid"))

#     # compile
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model


# --------------------------------------------- execute --------------------------------------------------- #
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
        #Validation set
        val_feats = np.array(feats[i*fold_size:(i+1)*fold_size])
        val_labels = np.array(labels[i*fold_size:(i+1)*fold_size])

        # Training set
        train_feats = np.array(feats[:i*fold_size] + feats[(i+1)*fold_size:])
        train_labels = np.array(labels[:i*fold_size] + labels[(i+1)*fold_size:])


        model_name = config["MODEL_NAME"]
        # ----------------------------- LR & SVM ------------------------------
        if model_name in ["LR", "SVM"]:
            pipeline = SklearnModelPipelineClassifier(config["MODEL_NAME"], 
                                                      config["common"]["max_iter"]
                                                    )
            print(f"\n\t*** TRAINING AND CROSS VALIDATION: {model_name} ***\t")
            classifier= pipeline.train(train_feats, train_labels)
            val_preds = classifier.classify_many(val_feats)

        # ---------------------------- LSTM -----------------------------------
        elif model_name =="LSTM":
            lstm_classifier = LSTMClassifier(config["LSTM"]["num_words"],
                                             config["LSTM"]["max_length"],
                                             config["LSTM"]["embed_dim"],
                                             config["LSTM"]["lstm_units"]
                                            )
            print(f"\n\t*** TRAINING AND CROSS VALIDATION: {model_name} ***\t")

            lstm_model, _ = lstm_classifier.train(train_feats, train_labels, 
                                                  val_feats, val_labels, 
                                                  config["LSTM"]["epoch"], 
                                                  config["LSTM"]["batch_size"]
                                                )
            val_preds = lstm_model(val_feats).numpy().flatten()
            val_preds = (val_preds >= 0.5).astype(int)


        # ------------------------- BERT -------------------------------
        elif model_name =="BERT":
            print("Inside the BERT model builder")
            bert_classifier=BERTForClassification(config["BERT"]["model_name"], 
                                                  config["BERT"]["seq_length"]
                                                )
            _= bert_classifier.train(train_feats, train_labels, 
                            val_feats, val_labels, 
                            config["BERT"]["epoch"],
                            config["BERT"]["batch_size"]
                            )

        else:
            raise ValueError(f"Unsupported Model name: {model_name}")
        
        # ---------------------------- k-fold acc (common) ---------------------

        acc = accuracy_score(val_labels, val_preds)
        (p, r, f, _) = precision_recall_fscore_support(y_pred=val_preds, y_true=val_labels, average='macro')
        print(f"Fold:{i + 1} -> Acc: {acc:.3f}, Precision: {p:.3f}, Recall: {r:.3f}, F-score: {f:.3f}\n")

        cv_scores.append((acc, p, r, f))
    cv_scores = np.mean(np.array(cv_scores), axis=0)
    return cv_scores
