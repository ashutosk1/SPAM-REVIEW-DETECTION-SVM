
import pandas as pd
import string
from collections import Counter

# nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# lstm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder


def LoadPreprocess(config):

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

    with open(config["REVIEW_TEXT_PATH"]) as f:
        review_df = pd.read_csv(f, encoding='latin-1', delimiter = "\t")
    
    # Drop Duplicates and modify the label names to be more readable. 
    review_df = review_df.drop_duplicates()
    review_df.loc[review_df["LABEL"] =="__label1__", "LABEL"] = "fake"
    review_df.loc[review_df["LABEL"] =="__label2__", "LABEL"] = "real"

    # Preprocesses a text string by performing lowercase conversion, punctuation removal, lemmatization, and stop word removal.
    review_df["REVIEW_TEXT"] = review_df["REVIEW_TEXT"].apply(lambda text:preprocess_text(text, config))

    if "REVIEW_TITLE" in config["FEATURES_LIST"]:
            review_df["REVIEW_TITLE"] = review_df["REVIEW_TITLE"].apply(lambda text:preprocess_text(text, config))

    # Encode the labels for Classification.
    label_encoder = LabelEncoder()
    review_df["LABEL"] = label_encoder.fit_transform(review_df["LABEL"])

    # Generate Feature Vector for different columns in the `feature_list`.
    review_df["FEATURE_VECTOR"] = review_df.apply(lambda row: [create_feature_vector(row, config["FEATURES_LIST"])], axis=1)
    review_df["FEATURE_VECTOR"] = review_df['FEATURE_VECTOR'].apply(lambda x : x[0])

    return review_df



def preprocess_text(text:str, config):
    """ Preprocesses a text string by performing lowercase conversion, punctuation removal, lemmatization, and stop word removal.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans({key: None for key in string.punctuation}))  

    tokens = word_tokenize(text)       # Tokenize text
    filtered_tokens = [token for token in tokens if token not in stop_words]    # Filter token by rejecting tokens for stopwords
    filtered_tokens = [WordNetLemmatizer().lemmatize(token) for token in filtered_tokens]     # Lemmatize Tokens: Change to base form
    
    if config["MODEL_NAME"]!="LSTM":  
        if config["common"]["ngrams"]==1:
            filtered_tokens = filtered_tokens
        elif config["common"]["ngrams"]==2:
            filtered_tokens.extend([' '.join(l) for l in nltk.bigrams(filtered_tokens)]) 
        elif config["common"]["ngrams"]==3:
            filtered_tokens.extend([' '.join(l) for l in nltk.trigrams(filtered_tokens)])
        else:
            raise ValueError(f"Invalid value of ngram. Accepted values are (1, 2, 3) only.")
        return filtered_tokens
    
    else:
        if config["common"]["ngrams"]==1:
            filtered_tokens = filtered_tokens
        else:
            print("[WARNING] Ngram >1 Tokenization for LSTM is not implemented yet!")
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



def lstm_preprocess(preprocessed_review_df, num_words, max_length):
    """
    Preprocessing for the LSTM Input Layer.
    """
    tokenizer = Tokenizer(num_words = num_words)
    tokenizer.fit_on_texts(preprocessed_review_df["FEATURE_VECTOR"].apply(lambda row:[key for key in row.keys()]))
    preprocessed_review_df["FEATURE_VECTOR"] = tokenizer.texts_to_sequences(preprocessed_review_df["FEATURE_VECTOR"].apply(lambda row: [key for key in row.keys()]))
    preprocessed_review_df["FEATURE_VECTOR"] = pad_sequences(preprocessed_review_df["FEATURE_VECTOR"], maxlen=max_length).tolist()
    return preprocessed_review_df, tokenizer