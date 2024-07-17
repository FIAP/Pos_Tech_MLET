from collections import Counter
import itertools

import pandas as pd


def count_tokens(df: pd.DataFrame, token_col: str) -> dict:
    """Count frequency of tokens.

    Args:
        df (pd.DataFrame): Dataframe.
        token_col (str): Column name with tokens.

    Returns:
        dict: token as key and frequency as value.
    """
    counter = Counter(list(itertools.chain.from_iterable(df[token_col].to_list())))
    print(f"Tamanho do vocabulário total: {len(counter)}")
    return counter
