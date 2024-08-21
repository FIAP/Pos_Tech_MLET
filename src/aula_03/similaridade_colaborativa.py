from typing import List, Tuple

import pandas as pd
from scipy.stats import pearsonr
import random


class StatisticalRecommender:
    def __init__(self, ratings: pd.DataFrame):
        self.ratings = ratings.pivot_table(index="user_id", columns="movie_id", values="rating")

    def get_similar_items(self, item_id: int, top_n: int = 10) -> List[Tuple[int, int]]:
        similar_scores = {}

        for other_item in self.ratings.columns:
            if other_item != item_id:
                common_ratings = self.ratings[[item_id, other_item]].dropna()

                if len(common_ratings) > 1:
                    correlation, _ = pearsonr(common_ratings[item_id], common_ratings[other_item])
                    similar_scores[other_item] = correlation

        sorted_items = sorted(similar_scores.items(), key=lambda x: x[1], reverse=True)
        top_similar_items = [(item[0], item[1]) for item in sorted_items[:top_n]]

        return top_similar_items


# Exemplo de dados de ratings
ratings_data = {
    "user_id": [random.randint(1, 5) for _ in range(1000)],
    "movie_id": [random.randint(101, 105) for _ in range(1000)],
    "rating": [random.randint(1, 5) for _ in range(1000)],
}
ratings_df = pd.DataFrame(ratings_data)

# Criação da instância do recomendador
recommender = StatisticalRecommender(ratings=ratings_df)

# Exemplo de uso para encontrar itens similares ao item 101
similar_items = recommender.get_similar_items(item_id=101)
print("Itens similares ao item 101:", similar_items)
