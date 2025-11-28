from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.post("/receber-dados")
def receber_dados():
    dados = request.get_json()
    print("Dados recebidos:", dados)

    return jsonify({"status": "success", "mensagem": "Dados recebidos com sucesso!"})

if __name__ == "__main__":
    app.run(debug=True)
