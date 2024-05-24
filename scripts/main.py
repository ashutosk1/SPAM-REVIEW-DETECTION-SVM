from preprocess import LoadPreprocess
from model import trainer
import json

if __name__ == "__main__":


    # Read json
    with open("config.json", 'r') as f:
            config = json.load(f)
    
    # Preprocess the Dataset
    processed_review_df = LoadPreprocess(config)

    # Train Model and report accuracy
    _ = trainer(processed_review_df, config)
