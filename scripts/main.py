from preprocess import LoadPreprocess
from model import trainer
import constants

if __name__ == "__main__":
    
    # Access the constant variables
    review_text_path = constants.REVIEW_TEXT_PATH
    features_list    = constants.FEATURE_LIST
    model_name       = constants.MODEL_NAME
    max_iter         = constants.MAX_ITER
    ngrams           = constants.NGRAMS
    folds            = constants.FOLDS

    # Preprocess the Dataset
    processed_review_df = LoadPreprocess(review_text_path, features_list, ngrams)
    
    # for i in range(10):
    #     print(processed_review_df.iloc[i]["FEATURE_VECTOR"])
    #     print(100*"*")

    # Train Model and report accuracy
    _ = trainer(processed_review_df, model_name, max_iter, folds)