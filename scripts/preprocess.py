
import pandas as pd
import string
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder


def LoadPreprocess(review_path:str, features_list:list, ngrams:int):

    """
    Loads and preprocesses a review dataset file.

    Args:
        review_path (str): Path to the tab-delimited file containing reviews and labels.
        features_list (list): List of features to include in the feature vector.
            - "REVIEW_TEXT": Word count features from review text
            - "REVIEW_TITLE" (Optional)
            - "RATING" (Optional)
            - "PRODUCT_CATEGORY" (Optional)
            - "VERIFIED_PURCHASE" (Optional)
        ngrams (int): Supports unigram, bigram and trigram for tokenization. 

    Returns:
        pandas.DataFrame: A DataFrame containing feature vectors as additional column.

    Steps:
     - Preprocessing steps:
      1. Load data using pandas.
      2. Remove duplicates and adjust label names.
      3. Convert text to lowercase and remove punctuation.
      4. Lemmatize words with provided value of ngrams and remove stop words.
      5. Encode labels.
      6. Create feature vectors based on the features_list argument.

    """

    with open(review_path) as f:
        review_df = pd.read_csv(f, encoding='latin-1', delimiter = "\t")
    
    # Drop Duplicates and modify the label names to be more readable. 
    review_df = review_df.drop_duplicates()
    review_df.loc[review_df["LABEL"] =="__label1__", "LABEL"] = "fake"
    review_df.loc[review_df["LABEL"] =="__label2__", "LABEL"] = "real"

    # Preprocesses a text string by performing lowercase conversion, punctuation removal, lemmatization, and stop word removal.
    review_df["REVIEW_TEXT"] = review_df["REVIEW_TEXT"].apply(lambda text:preprocess_text(text, ngrams))

    if "REVIEW_TITLE" in features_list:
            review_df["REVIEW_TITLE"] = review_df["REVIEW_TITLE"].apply(lambda text:preprocess_text(text, ngrams))

    # Encode the labels for Classification.
    label_encoder = LabelEncoder()
    review_df["LABEL"] = label_encoder.fit_transform(review_df["LABEL"])

    # Generate Feature Vector for different columns in the `feature_list`.
    review_df["FEATURE_VECTOR"] = review_df.apply(lambda row: [create_feature_vector(row, features_list)], axis=1)
    review_df["FEATURE_VECTOR"] = review_df['FEATURE_VECTOR'].apply(lambda x : x[0])

    return review_df


def preprocess_text(text:str, ngrams:int):
    """ Preprocesses a text string by performing lowercase conversion, punctuation removal, lemmatization, and stop word removal.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans({key: None for key in string.punctuation}))  

    tokens = word_tokenize(text)       # Tokenize text
    filtered_tokens = [token for token in tokens if token not in stop_words]    # Filter token by rejecting tokens for stopwords
    filtered_tokens = [WordNetLemmatizer().lemmatize(token) for token in filtered_tokens]     # Lemmatize Tokens: Change to base form
    if ngrams==2:
        filtered_tokens.extend([' '.join(l) for l in nltk.bigrams(filtered_tokens)])  # Efficiently add bigrams
    elif ngrams==3:
        filtered_tokens.extend([' '.join(l) for l in nltk.trigrams(filtered_tokens)])
    else:
        raise ValueError(f"Invalid value of ngram : {ngrams}. Accepted values are (1, 2, 3) only.")
    return filtered_tokens


def create_feature_vector(row, features_list):
    """Create feature vectors based on the features_list argument.
    """
    feature_vec_list ={}
    if "REVIEW_TEXT" in features_list:
        feature_vec_list.update(dict(Counter(row["REVIEW_TEXT"])))

    if "REVIEW_TITLE" in features_list:
        feature_vec_list.update(dict(Counter(row["REVIEW_TITLE"])))

    if "RATING" in features_list:
        feature_vec_list.update({"R" : row["RATING"]})

    if "PRODUCT_CATEGORY" in features_list:
        feature_vec_list.update(dict(Counter(row["PRODUCT_CATEGORY"].lower().split())))

    if "VERIFIED_PURCHASE" in features_list:
        feature_vec_list.update({"VP": 1} if row["VERIFIED_PURCHASE"]=="Y" else {"VP" : 0})
    return feature_vec_list