from preprocess import LoadPreprocess
from model import trainer
import json
import argparse


def read_json(config_path):
        """
        Reads JSON configuration file from the Input path and returns a dict.
        """
        try:
                with open(config_path, "r") as f:
                        config = json.load(f)
                print(f"[INFO] Loading configuration file from {config_path}.")
                return config

        except FileNotFoundError:
                print("[ERROR] Configuration file not found at {config_path}")
                return None


def get_config_path_from_args():
        """
        Parses command-line arguments to get the path to the configuration file.
        """
        parser = argparse.ArgumentParser(description="Train a model based on a JSON configuration file.")
        parser.add_argument(
                "--config",
                dest="config_path",
                type=str,
                required=True,
                help="Path to the JSON configuration file.",
        )
        args = parser.parse_args()
        return args.config_path


def main(config_path):
        config = read_json(config_path)

        # Preprocess the Dataset
        processed_review_df = LoadPreprocess(config)

        # for i in range(10):
        #         print(processed_review_df.iloc[i]["FEATURE_VECTOR"])
        #         print(100*"*")

        # Train Model and report accuracy
        cv_scores_dict = trainer(processed_review_df, config)
        print(cv_scores_dict)



if __name__ =="__main__":
        main(get_config_path_from_args())
