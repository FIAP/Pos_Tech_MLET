import os
import asyncio
from typing import Callable, List
from string import Template

import pandas as pd
from sklearn.cluster import KMeans
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

from aula_08.prompts import SYSTEM_PROMPT, AGENT_PROMPT
from aula_08.schemas import AzureAIMessage


CURRENT_DIR = os.path.dirname(__file__)
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), ".env")
load_dotenv(dotenv_path)

API_KEY = os.environ.get("GPT4O_KEY", "")
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_4O_ENDPOINT", "")


def aplicar_algoritmo_agrupamento(dados: pd.DataFrame, num_clusters: int):
    # Criar uma cópia dos dados originais
    dados_copia = dados.copy()

    # Inicializar o algoritmo de agrupamento K-means
    kmeans = KMeans(n_clusters=num_clusters)

    # Ajustar o modelo aos dados
    kmeans.fit(dados_copia)

    # Adicionar as informações de cluster aos dados originais
    dados_copia['cluster'] = kmeans.labels_

    # Retornar os dados com as informações de cluster
    return dados_copia


def caracterizar_resposta_esperada(dados: pd.DataFrame, funcao_utilidade: Callable[..., float]):
    # Aplicar a função de utilidade aos dados
    resultados = [funcao_utilidade(dado) for dado in dados]

    # Caracterizar a resposta esperada com base nos resultados de utilidade
    resposta_esperada = []

    for resultado in resultados:
        if resultado >= 0.8:
            resposta_esperada.append("Muito satisfeito")
        elif resultado >= 0.6:
            resposta_esperada.append("Satisfeito")
        elif resultado >= 0.4:
            resposta_esperada.append("Neutro")
        elif resultado >= 0.2:
            resposta_esperada.append("Insatisfeito")
        else:
            resposta_esperada.append("Muito insatisfeito")

    # Retornar a resposta esperada
    return resposta_esperada


def tomar_decisao_consumo(dados: pd.DataFrame, resposta_esperada: List[str]):
    # Definir a função que utiliza o SLM externo para tomar a decisão de consumo
    async def llm_externo(prompt):
        client = AsyncAzureOpenAI(
            api_key = API_KEY,
            azure_endpoint = AZURE_ENDPOINT
        )
        user_message = AzureAIMessage(
            role="user",
            content=Template(AGENT_PROMPT).safe_substitute(prompt),
            name="plan"
        )
        system_message = AzureAIMessage(
            role="system",
            content=SYSTEM_PROMPT,
            name="plan_system"
        )
        response = await client.chat.completions.create(
            model="rcataldi-gpt4o",
            messages=[system_message.model_dump(), user_message.model_dump()]  #type: ignore
        )
        return response.choices[0].message.content

    # Verificar se o dado deve ser consumido com base no agrupamento e na qualificação das utilidades
    decisoes_consumo = []
    for i, dado in enumerate(dados):
        if resposta_esperada[i] == "Muito satisfeito" and asyncio.run(llm_externo(dado)):
            decisoes_consumo.append("Consumir")
        else:
            decisoes_consumo.append("Não consumir")

    # Retornar as decisões de consumo
    return decisoes_consumo