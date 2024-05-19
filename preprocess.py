
import pandas as pd
import string
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder


def LoadPreprocess(review_path:str, features_list:list):

    """
    Loads and preprocesses a review dataset file.

    Args:
        review_path (str): Path to the tab-delimited file containing reviews and labels.
        features_list (list): List of features to include in the feature vector.
            - "REVIEW_TEXT": Word count features from review text
            - "REVIEW_TITLE" (Optional)
            - "RATING" (Optional)
            - "PRODUCT_CATEGORY"
            - "VERIFIED_PURCHASE" 

    Returns:
        pandas.DataFrame: A DataFrame containing preprocessed reviews, labels, and feature vectors.

    Steps:

     - Preprocessing steps:
      1. Load data using pandas.
      2. Remove duplicates and adjust label names.
      3. Convert text to lowercase and remove punctuation.
      4. Lemmatize words and remove stop words.
      5. Encode labels.
      6. Create feature vectors based on the features_list argument.

    """

    with open(review_path) as f:
        review_df = pd.read_csv(f, encoding='latin-1', delimiter = "\t")
    
    # Drop Duplicates and modify the label names to be more readable. 
    review_df = review_df.drop_duplicates()
    review_df.loc[review_df["LABEL"] =="__label1__", "LABEL"] = 'fake'
    review_df.loc[review_df["LABEL"] =="__label2__", "LABEL"] = 'real'



    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    review_df["REVIEW_TEXT"] = review_df["REVIEW_TEXT"].str.lower()
    review_df["REVIEW_TEXT"] = review_df["REVIEW_TEXT"].str.translate(str.maketrans({key: None for key in string.punctuation}))
    review_df["REVIEW_TEXT"] = review_df["REVIEW_TEXT"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) \
                                                                                  for word in x.split()\
                                                                                  if word not in stop_words]))
    # Encode the labels for Classification
    label_encoder = LabelEncoder()
    review_df["LABEL"] = label_encoder.fit_transform(review_df["LABEL"])

    def create_feature_vector(row, features_list):
        feature_vec_list ={}

        if "REVIEW_TEXT" in features_list:
            feature_vec_list.update(dict(Counter(row["REVIEW_TEXT"].lower().split())))

        if "REVIEW_TITLE" in features_list:
            feature_vec_list.update(dict(Counter(row["REVIEW_TITLE"].lower().split())))

        if "RATING" in features_list:
            feature_vec_list.update({"R" : row["RATING"]})

        if "PRODUCT_CATEGORY" in features_list:
            feature_vec_list.update(dict(Counter(row["PRODUCT_CATEGORY"].lower().split())))

        if "VERIFIED_PURCHASE" in features_list:
            feature_vec_list.update({"VP": 1} if row["VERIFIED_PURCHASE"]=="Y" else {"VP" : 0})

        return feature_vec_list

    review_df["FEATURE_VECTOR"] = review_df.apply(lambda row: [create_feature_vector(row, features_list)], axis=1)
    review_df["FEATURE_VECTOR"] = review_df['FEATURE_VECTOR'].apply(lambda x : x[0])

    return review_df