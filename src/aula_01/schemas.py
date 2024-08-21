from typing import List, Optional

from pydantic import BaseModel


class ItemClick(BaseModel):
    item_id: str
    timestamp: float


class Item(BaseModel):
    id: str
    name: str
    price: float
    clicks: Optional[List[ItemClick]] = None
    sentiment_scores: Optional[List[int]] = None
    description: Optional[str] = None
