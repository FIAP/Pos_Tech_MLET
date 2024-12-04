from abc import ABC, abstractmethod

import pandas as pd
from datasets import load_dataset


class Dataset(ABC):

    @abstractmethod
    def load_pd_test_dataset(self, max_size: int = 20, **kwargs) -> pd.DataFrame:
        """Load test dataset.

        Args:
            max_size (int): Maximum quantity of rows to retrieve. Default to 20

        Returns:
            pd.DataFrame: Test dataset
        """
        pass


class HuggingFaceDataset(Dataset):

    def load_pd_test_dataset(self, max_size: int = 20, **kwargs) -> pd.DataFrame:
        """Load test dataset.

        Args:
            max_size (int): Maximum quantity of rows to retrieve. Default to 20

        Returns:
            pd.DataFrame: Test dataset
        """

        dataset = (
            load_dataset(
                **kwargs
            ).select(range(max_size)).to_pandas()
        )
        return dataset.rename(columns={"text": "inputs"})