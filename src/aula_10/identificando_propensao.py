import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def identificar_propensao(dados):
    # Separar os dados em variáveis de entrada (X) e variável de saída (y)
    X = dados.drop('propensao', axis=1)
    y = dados['propensao']

    # Dividir os dados em conjunto de treinamento e conjunto de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criar o pipeline com o scaler e o modelo de regressão com XGBoost
    model = Pipeline([
        ('scaler', MinMaxScaler()),  # Aplica o MinMaxScaler aos dados numéricos
        ('regressor', xgb.XGBRegressor())  # Modelo de regressão com XGBoost
    ])

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Avaliar o desempenho do modelo no conjunto de teste
    score = model.score(X_test, y_test)

    return model, score
