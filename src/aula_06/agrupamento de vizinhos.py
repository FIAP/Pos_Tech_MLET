import random

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from aula_06.schemas import ItemRelationship


def knn_clustering(data, k):
    # Separando as colunas relevantes para o agrupamento
    data = pd.DataFrame([vars(d) for d in data])
    features = ['idade', 'renda', 'cliques_no_item', 'compras_do_item', 'tempo_carrinho']
    X = data[features]

    # Normalizando os dados
    X_normalized = (X - X.mean()) / X.std()

    # Aplicando o algoritmo KNN
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_normalized)

    # Obtendo os k vizinhos mais próximos para cada ponto
    distances, indices = knn.kneighbors(X_normalized)

    return distances, indices


# Gerando aleatoriamente 100 usuários seguindo o modelo ItemRelationship
def main():
    users = []
    for i in range(100):
        user = ItemRelationship(
            item_id=i,
            user_id=random.randint(1, 100),
            idade=random.randint(14, 65),
            profissao=random.choice(['Estudante', 'Engenheiro', 'Professor', 'Médico']),
            renda=random.uniform(1000, 5000),
            cliques_no_item=random.randint(0, 100),
            compras_do_item=random.randint(0, 10),
            tempo_carrinho=random.randint(0, 60)
        )
        users.append(user)

    # Aplicando o clustering de knn sobre os usuários
    distance, indexes = knn_clustering(users, k=5)
    return distance, indexes


if __name__ == '__main__':
    result = main()
    print("Distâncias: ", result[0])
    print("Índices: ", result[1])
