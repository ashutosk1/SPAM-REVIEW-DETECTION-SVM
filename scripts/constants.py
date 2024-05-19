REVIEW_TEXT_PATH = "amazon_reviews.txt"
FEATURE_LIST = ["REVIEW_TEXT", "VERIFIED_PURCHASE"]     #["REVIEW_TEXT", "REVIEW_TITLE", "RATING", "PRODUCT_CATEGORY", "VERIFIED_PURCHASE"]
MODEL_NAME = "LogisticRegression"   #{"LogisticRegression", "SVM"}
TEST_SIZE = 0.2
MAX_ITER = 10000
NGRAMS = 1                              #{1, 2, 3}