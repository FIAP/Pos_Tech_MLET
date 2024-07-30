from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing import TextPreprocessing


class TextCleaner(BaseEstimator, TransformerMixin, TextPreprocessing):
    def __init__(self, **kwargs):
        """Initialize TextCleaner"""
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(lambda x: self.preprocess_text(text=x, **self.kwargs))
