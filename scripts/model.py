from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk.classify import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import pandas as pd
import numpy as np
import pprint

# Metrics
from sklearn.metrics import classification_report

# Local
from preprocess import LoadPreprocess

def train_model(processed_review_df:pd.DataFrame, model_name:str, test_size_frac:float, max_iter:int):
    
    """
    Trains a classification model on a DataFrame containing processed reviews - with feature vector and labels.

    Args:
        processed_review_df (pd.DataFrame): DataFrame containing processed review data.
        model_name (str): The name of the classification model to train. Supports
            "LogisticRegression" and "SVM".
        test_size_frac (float): The fraction of the data to use for the test set.
        max_iter (int): The maximum number of iterations for the chosen model.

    Returns:
        SklearnClassifier: The trained classification model.
    """
    
    processed_review_df_feats = np.array(processed_review_df["FEATURE_VECTOR"])
    processed_review_df_labels= processed_review_df["LABEL"].to_numpy()

    split_list_test_train_feat_labels = train_test_split(processed_review_df_feats, 
                                                         processed_review_df_labels, 
                                                         test_size=test_size_frac,
                                                         shuffle=True
                                                    )
    train_feat_vectors = split_list_test_train_feat_labels[0]
    test_feat_vector   = split_list_test_train_feat_labels[1]
    train_labels       = split_list_test_train_feat_labels[2]
    test_labels        = split_list_test_train_feat_labels[3]

    combined_train_features_and_labels = list(zip(train_feat_vectors, train_labels))

    # Define the model based on model_name
    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=max_iter)
        pipeline =  Pipeline([('clf', model)])
    elif model_name == "SVM":
        model = LinearSVC()
        pipeline =  Pipeline([('svc', model)])
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Train the model on the combined features and labels    
    classifier = SklearnClassifier(pipeline).train(combined_train_features_and_labels)

    # Print the classification report
    cf_report = classification_report(y_true=test_labels, 
                                   y_pred=classifier.classify_many(test_feat_vector),
                                   output_dict=False)
    print(cf_report)

    return classifier

