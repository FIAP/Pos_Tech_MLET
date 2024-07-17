"""Module responsible for preprocessing methods."""

import re
from typing import List
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk import word_tokenize


class TextPreprocessing:
    """class responsible for TextPreprocessing methods."""

    def preprocess_text(
        self,
        text: str,
        apply_lower: bool = False,
        remove_ponctuation: bool = False,
        apply_unidecode: bool = False,
        remove_stopwords: bool = False,
        apply_stemming: bool = False,
        apply_lemmitization: bool = False,
        lemmatizer: object = None,
    ) -> str:
        """Preprocess text.

        Args:
            text (str): Text.
            apply_lower (bool, optional): Lowercase the text or not. Defaults to False.
            remove_ponctuation (bool, optional): Remove ponctuatuin or not. Defaults to False.
            apply_unidecode (bool, optional): Apply unidecode or not. Defaults to False.
            remove_stopwords (bool, optional): Remove stopwords or not. Defaults to False.
            apply_stemming (bool, optional): Apply stemming or not. Defaults to False.
            apply_lemmitization (bool, optional): Apply lemmitization or not. Defaults to False.
            lemmatizer (object, optional): Which lemmatizer to use. Defaults to None.

        Returns:
            str: Text preprocessed.
        """
        if apply_lemmitization and lemmatizer is None:
            raise AssertionError(
                "You should provide a lemmatizer when flag apply_lemmitization is True."
            )
        text = text.lower() if apply_lower else text
        text = re.sub(r"[^\w\s]", " ", text) if remove_ponctuation else text
        text = unidecode(text) if apply_unidecode else text
        text = self.remove_stopwords(text) if remove_stopwords else text
        text = self.apply_stemming(text) if apply_stemming else text
        text = (
            self.apply_lemmitization(text, lemmatizer)
            if apply_lemmitization and lemmatizer is not None
            else text
        )
        return text

    def get_stopwords(self) -> List[str]:
        """Get stopwords.

        Returns:
            List[str]: List of stopwords.
        """
        return stopwords.words("portuguese")

    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords.

        Args:
            text (str): Text.

        Returns:
            str: Text without stopwords.
        """
        stop_words = set(self.get_stopwords())
        tokens = word_tokenize(text)
        valid_tokens = [token for token in tokens if token not in stop_words]
        return " ".join(valid_tokens)

    def apply_stemming(self, text: str) -> str:
        """Apply stemming.

        Args:
            text (str): Text.

        Returns:
            str: Stemmed text.
        """
        stemmer = RSLPStemmer()
        tokens = word_tokenize(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        return " ".join(stemmed_tokens)

    def apply_lemmitization(self, text: str, lemmatizer) -> str:
        """Apply lemmitization.

        Args:
            text (str): Text.
            lemmatizer (object): Lemmatizer.

        Returns:
            str: Lemmatized text.
        """
        doc = lemmatizer(text)
        token_lemmatized = [token.lemma_ for token in doc]
        return " ".join(token_lemmatized)
