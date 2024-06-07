
# Spam Review Classification with Transformers

This project focuses on classifying spam reviews from Amazon Review's dataset using various machine learning models, including Logistic Regression (LR), Support Vector Machine (SVM), and Long Short-Term Memory (LSTM) networks. A unified and config-driven command-line interface (CLI) tool is provided to streamline experimentation on preprocessing by the addition of different features, training, and k-fold cross-validation.



## Features

- Preprocessing of text data - Feature generation with tokenization, nltk-based text-cleaning, n-gram tokenization, Word2Vec tokenization.
- Implementation of BERT model benchmarked with LR, SVM, and LSTM.
- Model evaluation and comparison using cross-validation. 
- Unified and config-driven CLI tool for seamless experimentation.





## Usage/Examples

1. Clone the repository
```
git clone https://github.com/ashutosk1/Spam-Review-Analysis.git
```

3. Install the required packages 
```
pip install -r requirements.txt
```

5. Modify the configuration file. Ensure the path to the dataset is correctly filled. Select the features and model as listed in `_all_feature_lists` and `all_models_list` respectively. Save and get the path of `config.json`.
```
{
    "FOLD": 5,
    "MODEL_NAME": "BERT"  , 
    "REVIEW_TEXT_PATH": "/home/ashutosk/DL_SPAM_REVIEW_CLASSIFICATION/amazon_reviews.txt",
    "FEATURES_LIST": ["REVIEW_TEXT", "REVIEW_TITLE", "VERIFIED_PURCHASE", "RATING"],
    "common": {
        "max_iter": 10000,
        "ngrams": 1
    },
    "LSTM": {
        "num_words": 10000,
        "max_length": 128,
        "embed_dim": 128,
        "lstm_units": 128,
        "epoch"     : 1,
        "batch_size": 16
    },
    "BERT": {
        "seq_length" : 128,
        "model_name" : "bert-base-uncased",
        "epoch"     : 5,
        "batch_size": 16
    },
    "_all_models_list" :"{BERT, LSTM, LR, SVM}",
    "_all_feature_lists" : "{REVIEW_TEXT, VERIFIED_PURCHASE, RATING, REVIEW_TITLE, PRODUCT_CATEGORY}"
}
```
4. Train the model and get the cross-validation metrics.

```
cd ./scripts
python3 main.py --config /path/to/config
```


## Project Structure
```
├── Exploration.ipynb
├── README.md
├── amazon_reviews.txt
├── requirements.txt
└── scripts
    ├── bert.py
    ├── config.json
    ├── lstm.py
    ├── main.py
    ├── model.py
    ├── preprocess.py
    └── sklearn_models.py
```


## Results
| k-fold accuracy score (macro, k=5)                                     |         |       |       |   
|------------------------------------------------------------------------|---------|-------|-------|
| features                                                               | ngrams  | lr    | svm   |   
| review_title                                                           | 1       | 0.620 | 0.596 |  
|                                                                        | 2       | 0.635 | 0.626 |   
|                                                                        | 3       | 0.636 | 0.624 |   
| review_text, verified_purchase, rating, review_title, product_category | 1       | 0.797 | 0.757 |   
|                                                                        | 2       | 0.810 | 0.802 |
|                                                                        | 3       | 0.811 | 0.807 |   
| review_text, verified_purchase                                         | 1       | 0.794 | 0.754 |   
|                                                                        | 2       | 0.812 | 0.801 | 
|                                                                        | 3       | 0.812 | 0.807 |


## Dataset
https://drive.google.com/file/d/1q9ioiENrQvP9Fnu0jzItdcxLQeTzWVUU/view?usp=sharing
