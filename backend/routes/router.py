import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.service.serviceAI import call_models
app = Flask(__name__)
CORS(app)

@app.post("/receber-dados")
def receber_dados():
    dados = request.get_json()
    responses_models = call_models(dados)
    return jsonify({"status": "success", "mensagem": "Dados recebidos com sucesso!", "data": responses_models})

if __name__ == "__main__":
    app.run(debug=True)
