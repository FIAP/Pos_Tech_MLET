from prometheus_client import start_http_server, Summary, Counter, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

from .model import load_model, predict

model = load_model("saved_model/model_transfer_learning.pth")

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
INFERENCE_COUNT = Counter('inference_count', 'Total number of inferences made')
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy over time')

# Função para inferência com monitoramento
@REQUEST_TIME.time()
def process_request(image):
    INFERENCE_COUNT.inc()
    prediction = predict(model, image)
    # Exemplo de métrica para monitoramento de acurácia (valor fixo para o exemplo)
    accuracy = 0.9  # Supondo que esta é uma métrica fixa ou calculada de alguma forma
    MODEL_ACCURACY.set(accuracy)
    return prediction
