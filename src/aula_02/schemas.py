from typing import Dict, Optional

from pydantic import BaseModel


class UserProfile(BaseModel):
    user_id: int
    name: str
    age: int
    gender: str
    profession: str


class UserInteractions(BaseModel):
    user_id: int
    interactions: Dict[int, int]


class Item(BaseModel):
    id: str
    name: str
    price: float
    clicks: Optional[int] = None
    sentiment_scores: Optional[float] = None
    description: Optional[str] = None
