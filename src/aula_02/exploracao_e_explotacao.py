from typing import List

import pandas as pd
from scipy import stats

from aula_02.schemas import UserInteractions, UserProfile, Item
from aula_01.recomendacao_popularidade import Recommender


class RecommenderForExploitation(Recommender):

    def __init__(self, data: List[Item]):
        self.data = data

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
        return item.clicks

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
        return item.sentiment_scores


class Exploiter:
    def __init__(self, movies_metadata: pd.DataFrame, ratings: pd.DataFrame):
        self.movies_metadata = movies_metadata
        self.ratings = ratings

    @staticmethod
    def __dataframe_to_items(data: pd.DataFrame) -> List[Item]:
        items = []
        for _, row in data.iterrows():
            item = Item(
                id=row['id'],
                name=row['title'],
                price=row['price'] or 0,
                clicks=row['vote_count'],
                sentiment_scores=row['vote_average'],
                description=row['description'] or ""
            )
            items.append(item)
        return items

    @staticmethod
    def __items_to_dataframe(items: List[Item]) -> pd.DataFrame:
        data = []
        for item in items:
            data.append({
                'id': item.id,
                'title': item.name,
                'vote_count': item.clicks,
                'vote_average': item.sentiment_scores
            })
        return pd.DataFrame(data)

    async def get_popular_movies(self, top_n: int = 10) -> pd.DataFrame:
        data = self.__dataframe_to_items(self.movies_metadata)
        recommender = RecommenderForExploitation(data)
        popular_movies = await recommender()
        popular_movies = self.__items_to_dataframe(popular_movies)  # type: ignore
        return popular_movies.head(top_n)

    def recommend_by_exploration(self, user_interactions: UserInteractions) -> List[int]:
        normalized_values = stats.zscore(user_interactions.interactions.values())
        normalized_values = normalized_values.tolist()
        weights = {
            item_id: normalized_value
            for (item_id, _), normalized_value in zip(user_interactions.interactions.items(), normalized_values)
        }

        return list(weights.keys())

    def exploit_similar_profiles(
        self, user_profile: UserProfile, all_user_profiles: pd.DataFrame
    ) -> List[int]:
        similar_users = all_user_profiles[
            (all_user_profiles["profession"] == user_profile.profession)
            & (abs(all_user_profiles["age"] - user_profile.age) <= 5)
        ]
        similar_user_ids = similar_users["user_id"].tolist()
        similar_user_ratings = self.ratings[self.ratings["user_id"].isin(similar_user_ids)]
        recommended_movies = similar_user_ratings.groupby("movie_id")["rating"].mean().reset_index()
        top_movies = (
            recommended_movies.sort_values(by="rating", ascending=False)
            .head(10)["movie_id"]
            .tolist()
        )
        return top_movies


def main():
    movies_metadata = pd.read_csv("movies_metadata.csv")
    ratings = pd.read_csv("ratings.csv")
    recommender = Exploiter(movies_metadata, ratings)

    popular_movies = recommender.get_popular_movies()
    user_interactions = UserInteractions(user_id=1, interactions={101: 5, 102: 3, 103: 2})
    explored_recommendations = recommender.recommend_by_exploration(user_interactions)

    user_profile = UserProfile(
        user_id=1, name="John Doe", age=25, gender="male", profession="engineer"
    )  # Adjusted for correct type
    all_user_profiles = pd.DataFrame()  # Load or simulate user profiles DataFrame

    exploited_recommendations = recommender.exploit_similar_profiles(
        user_profile, all_user_profiles
    )

    print("Popular Movies:", popular_movies)
    print("Explored Recommendations:", explored_recommendations)
    print("Exploited Recommendations:", exploited_recommendations)


if __name__ == "__main__":
    main()
