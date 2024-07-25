from functools import partial
from typing import Union
import pickle

import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from src.cleaner import TextCleaner

random_state = 42
model_choices = ["SVC", "GradientBoostingClassifier", "LogisticRegression"]


class Experiment:

    def get_classifier_search_space(self, trial: optuna.trial.Trial, classifier_name: str) -> dict:
        """Get classifier search space.

        Args:
            trial (optuna.trial.Trial): Trial.
            classifier_name (str): Classifier name.

        Raises:
            ValueError: When classifier name not in suported models.

        Returns:
            dict: Search space.
        """
        if classifier_name == "SVC":
            search_space = {
                "params": {"C": trial.suggest_float("svc_c", 1e-10, 1e10, log=True)},
                "model": SVC(gamma="auto", random_state=random_state),
            }
        elif classifier_name == "GradientBoostingClassifier":
            search_space = {
                "params": {
                    "max_features": trial.suggest_categorical("max_features", ["log2", "sqrt"]),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 1, step=0.1),
                    "max_depth": trial.suggest_int("max_depth", 3, 6),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 4, 6),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 100),
                },
                "model": GradientBoostingClassifier(random_state=random_state),
            }
        elif classifier_name == "LogisticRegression":
            search_space = {
                "params": {"C": trial.suggest_float("C", 1e-10, 1e10, log=True)},
                "model": LogisticRegression(random_state=random_state),
            }
        else:
            model_choices = ",".join(model_choices)
            raise ValueError(f"Model {classifier_name} not suported. Suported models are: {model_choices}")
        return search_space

    def objective(self, trial: optuna.trial.Trial, X: Union[pd.Series, np.array], y: Union[pd.Series, np.array], vectorize: bool = True, prefix: str = "") -> float:
        """Objective function.

        Args:
            trial (optuna.trial.Trial): Trial.
            X (Union[pd.Series, np.array]): Features.
            y (Union[pd.Series, np.array]): Target variable.
            vectorize (bool, optional): If to vectorize or not. Defaults to True.

        Returns:
            float: F1 score.
        """
        classifier_name = trial.suggest_categorical(
            "classifier", model_choices
        )
        search_space_classifier = self.get_classifier_search_space(trial, classifier_name=classifier_name)
        params = search_space_classifier["params"]
        model = search_space_classifier["model"].set_params(**params)

        if vectorize:
            vectorizer_params = {"ngram_range": (1, trial.suggest_int("ngram_range", 1, 2))}
            preprocessing_params = {
                "apply_lower": trial.suggest_int("apply_lower", 0, 1),
                "remove_ponctuation": trial.suggest_int("remove_ponctuation", 0, 1),
                "remove_numbers": trial.suggest_int("remove_numbers", 0, 1),
                "apply_unidecode": trial.suggest_int("apply_unidecode", 0, 1),
                "remove_stopwords": trial.suggest_int("remove_stopwords", 0, 1),
                "remove_short_tokens": trial.suggest_int("remove_short_tokens", 0, 1),
                "limit_consecutive_chars": trial.suggest_int("limit_consecutive_chars", 0, 1),
            }
            pipeline = Pipeline(
                [
                    ("preprocessing", TextCleaner(**preprocessing_params)),
                    ("vectorizer", CountVectorizer(**vectorizer_params)),
                    ("classifier", model),
                ]
            )
        else:
            pipeline = model
        score = sklearn.model_selection.cross_val_score(
            pipeline, X, y, n_jobs=-1, cv=3, scoring="f1"
        )
        pipeline.fit(X, y)
        with open(f"../models/{prefix}_clf_exp_{trial.number}.pickle", "wb") as fout:
            pickle.dump(pipeline, fout)
        f1 = score.mean()
        return f1

    def run_experiment(self, X: Union[pd.Series, np.array], y: Union[pd.Series, np.array], n_trials: int, vectorize: bool = True, prefix: str = "") -> optuna.study.Study:
        """Run experiment.

        Args:
            X (Union[pd.Series, np.array]): Features.
            y (Union[pd.Series, np.array]): Target variable.
            vectorize (bool, optional): If to vectorize or not. Defaults to True.
            n_trials (int): Number of trials.

        Returns:
            optuna.study.Study
        """
        objective = partial(self.objective, X=X, y=y, vectorize=vectorize, prefix=prefix)
        study = optuna.create_study(study_name=f"{prefix}_study", direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        study.trials_dataframe().to_csv(f"../models/{prefix}_clf_exp_results.csv")
        return study
