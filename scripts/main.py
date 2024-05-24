from preprocess import LoadPreprocess
from model import trainer
import json
import pprint


def read_json(config_path):
        with open(config_path, "r") as f:
              config = json.load(f)
        return config



def main(config_path):
        config = read_json(config_path)
        # Preprocess the Dataset
        processed_review_df = LoadPreprocess(config)
        # Train Model and report accuracy
        cv_scores_dict = trainer(processed_review_df, config)
        pprint.pprint(cv_scores_dict)



if __name__ =="__main__":
        config_path = "config.json"
        main(config_path)
