from pydantic import BaseModel


class ItemRelationship(BaseModel):
    item_id: int
    user_id: int
    idade: int
    profissao: str
    renda: float
    cliques_no_item: int
    compras_do_item: int
    tempo_carrinho: int
