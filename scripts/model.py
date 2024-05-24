from sklearn.pipeline import Pipeline
from nltk.classify import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import pandas as pd
import numpy as np
from random import shuffle

import warnings
warnings.filterwarnings("ignore")



def trainer(processed_review_df, model_name, max_iter, fold):
    """
    Trains and evaluates the classification model using cross-validation. 
    Steps:
    ** Extract Feature Vectors and Labels from preprocessed review dataset.**
    ** Build Classification Model with `model_name` and `max_iter` as the params.**
    ** Get Cross-Validation score by fitting the model on the train set and evaluating on validation set.** 
    """
    processed_review_df_feats = np.array(processed_review_df["FEATURE_VECTOR"])
    processed_review_df_labels= processed_review_df["LABEL"].to_numpy()

    pipeline = get_model_pipeline(model_name, max_iter)
    cv_scores = get_cross_validation_acc(pipeline, processed_review_df_feats, processed_review_df_labels, fold)                     



def get_model_pipeline(model_name, max_iter):
    """
    Constructs and returns a classification model pipeline to br trained with `SklearnClassifier`.
    """
    # Define the model based on model_name
    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=max_iter)
        pipeline =  Pipeline([('clf', model)])
    elif model_name == "SVM":
        model = LinearSVC()
        pipeline =  Pipeline([('svc', model)])
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    return pipeline



def get_cross_validation_acc(pipeline, feats, labels, fold):
    """
    Performs k-fold cross-validation and prints performance metrics for each fold.
    """

    assert len(feats) == len(labels)
    fold_size = len(labels)//fold
    cv_scores = []

    # Shuffle
    merged_feats_and_labels = list(zip(feats, labels))
    shuffle(merged_feats_and_labels)
    feats, labels = zip(*merged_feats_and_labels)

    for i in range(fold):
        # Validation set
        val_feats = feats[i*fold_size:(i+1)*fold_size]
        val_labels = labels[i*fold_size:(i+1)*fold_size]

        # Training set
        train_feats = feats[:i*fold_size] + feats[(i+1)*fold_size:]
        train_labels = labels[:i*fold_size] + labels[(i+1)*fold_size:]    

        # Clone pipeline for a fresh start
        pipeline_clone = clone(pipeline)
    
        classifier = SklearnClassifier(pipeline_clone).train(list(zip(train_feats, train_labels)))
        val_preds = classifier.classify_many(val_feats)
        acc = accuracy_score(val_labels, val_preds)
        (p, r, f, _) = precision_recall_fscore_support(y_pred=val_preds, y_true=val_labels, average='macro')

        print(f"fold:{i+1}, acc:{acc:.3f}, precision:{p:.3f}, recall:{r:.3f}, f-score:{f:.3f}")
        cv_scores.append((acc, p, r, f))

    return np.mean(np.array(cv_scores), axis=0)