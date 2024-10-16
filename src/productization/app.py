import io
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Summary, Counter, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from PIL import Image
from business_rules.log_nn import process_request

app = Flask(__name__)


# Rota para inferência
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    try:
        image = Image.open(io.BytesIO(file.read()))
        prediction = process_request(image)
        return jsonify({'prediction': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Rota para expor métricas do Prometheus
@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    start_http_server(8000)  # Expor as métricas na porta 8000
    app.run(host='0.0.0.0', port=5000)
