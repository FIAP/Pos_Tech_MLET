"""
Este script define a classe `Recommender` que é usada para recomendar itens com base em sua popularidade.

A popularidade de um item é calculada como a média do score de cliques e do score de sentimentos do item.

O score de cliques de um item é calculado como a soma do inverso do timestamp de cada clique no item.

O score de sentimentos de um item é calculado como a média dos sentimentos associados ao item.

A classe `Recommender` recebe uma lista de itens como entrada e fornece um método para calcular a popularidade de cada item e retornar uma lista de itens ordenados por sua popularidade.

Este script também define a função `main` que gera uma lista de itens com dados aleatórios e usa a classe `Recommender` para calcular a popularidade de cada item e imprimir a lista de itens ordenados por popularidade.
"""

import asyncio
from typing import List

import numpy as np
import pandas as pd

from aula_01.schemas import Item, ItemClick


class Recommender:
    """
    A class used to recommend items based on their popularity.

    Attributes
    ----------
    data : List[Item]
        A list of items to be recommended.

    Methods
    -------
    calculate_click_score(item: Item) -> float:
        Calculates the click score for a given item.

    calculate_sentiment_score(item: Item) -> float:
        Calculates the sentiment score for a given item.

    calculate_popularity() -> List[Item]:
        Calculates the popularity score for each item and returns a list of items sorted by their popularity score.
    """

    def __init__(self, data: List[Item]):
        self.data = data

    async def __call__(self) -> list[Item]:
        """
        Calculates the popularity score for each item and returns a list of items sorted by their popularity score.

        Returns
        -------
        List[Item]
            A list of items sorted by their popularity score.
        """
        if not self.data:
            raise ValueError("No data provided to the recommender")

        # Calculate scores for each item
        scores = []
        for item in self.data:
            click_score = self.calculate_click_score(item)
            sentiment_score = self.calculate_sentiment_score(item)
            combined_score = (click_score + sentiment_score) / 2
            scores.append((item, combined_score))

        # Sort items by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return the sorted list of items
        return [item for item, score in scores]

    @staticmethod
    def calculate_click_score(item: Item) -> float:
        """
        Calculates the click score for a given item.

        Parameters
        ----------
        item : Item
            The item for which the click score is to be calculated.

        Returns
        -------
        float
            The click score of the item.
        """
        # Check if the item has clicks
        if item.clicks is None:
            return 0

        # Convert the list of clicks to a DataFrame
        df = pd.DataFrame([click.model_dump() for click in item.clicks])

        # Calculate the score as the sum of the inverse of the timestamp
        score = (1 / df["timestamp"]).sum()

        return score

    @staticmethod
    def calculate_sentiment_score(item: Item) -> float:
        """
        Calculates the sentiment score for a given item.

        Parameters
        ----------
        item : Item
            The item for which the sentiment score is to be calculated.

        Returns
        -------
        float
            The sentiment score of the item.
        """
        # Check if the item has sentiment scores
        if item.sentiment_scores is None:
            return 0

        # Calculate the mean of the sentiment scores
        sentiment_score = pd.Series(item.sentiment_scores).mean()

        return sentiment_score


async def main():
    # Generate a list of random items
    items = [
        Item(
            id=str(i),
            name=f"Item {i}",
            price=np.random.uniform(1, 100),
            clicks=[
                ItemClick(item_id=str(i), timestamp=np.random.randint(1, 1000))
                for j in range(np.random.randint(1, 10))
            ],
            sentiment_scores=[np.random.randint(-10, 10) for _ in range(np.random.randint(0, 1))],
            description=f"Description for Item {i}",
        )
        for i in range(10)
    ]

    recommender = Recommender(items)
    print(await recommender())


if __name__ == "__main__":
    asyncio.run(main())
