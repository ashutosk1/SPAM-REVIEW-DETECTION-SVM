from sklearn.pipeline import Pipeline
from nltk.classify import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class SklearnModelPipelineClassifier:
    """
    sklearn-based lassification model pipeline for training with `SklearnClassifier`.
    """

    def __init__(self, model_name, max_iter):
        """
        Initialize with the model_name and max_iter.
        """
        self.model_name = model_name
        self.max_iter = max_iter
        self.pipeline = None  

    def get_pipeline(self):
        """
        Constructs and returns the classification model pipeline.
        """
        if self.pipeline is None:
            if self.model_name == "LR":
                model = LogisticRegression(max_iter=self.max_iter)
                self.pipeline = Pipeline([('clf', model)])
            elif self.model_name == "SVM":
                model = LinearSVC()
                self.pipeline = Pipeline([('svc', model)])
            else:
                raise ValueError(f"Invalid model name: {self.model_name}")
        return self.pipeline

    def train(self, feats, labels):
        """Trains the model."""
        pipeline = self.get_pipeline()
        classifier = SklearnClassifier(pipeline).train(list(zip(feats, labels)))
        return classifier