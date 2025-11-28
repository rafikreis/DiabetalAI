# 🩺 DiabetalAI

> Projeto acadêmico de Inteligência Artificial focado em diabetes.

## 📖 Sobre o Projeto

O **DiabetalAI** é uma aplicação desenvolvida como projeto de faculdade que utiliza Inteligência Artificial para auxiliar no contexto de diabetes (predição, monitoramento ou análise). O sistema é composto por um backend em Python e um frontend simples em HTML.

---

## 🚀 Tecnologias Utilizadas

*   **Backend:** Python
*   **IA/Data Science:** Pandas, Scikit-learn, TensorFlow, etc...
*   **Frontend:** HTML/CSS/JS

---

## ⚙️ Pré-requisitos

Antes de começar, certifique-se de ter o **Python** instalado em sua máquina.

## 📦 Instalação e Configuração

Siga os passos abaixo para configurar o ambiente de desenvolvimento:

1. **Navegue até o diretório do projeto:**
   Certifique-se de estar na pasta raiz `DiabetalAI`.

2. **Instale as dependências do Backend:**
   Acesse a pasta do backend e instale os pacotes listados no `requirements.txt`.

   ```bash
   cd backend
   pip install -r requirements.txt

**Dica**: É recomendável utilizar um ambiente virtual (venv) antes de instalar as dependências para evitar conflitos.

## ⚡ Como Rodar o Projeto

Para utilizar a aplicação, você precisará iniciar o servidor backend e depois abrir a interface.

1. **Iniciar o Servidor (Backend)**
    Partindo da raiz do projeto (DiabetalAI), execute os seguintes comandos:

    ```bash
    cd backend/routes
    python router.py
    
O terminal indicará que o servidor está rodando (geralmente em localhost ou 127.0.0.1). Mantenha este terminal aberto.

2. **Acessar a Interface (Frontend)**

    Com o backend rodando:
    Navegue até a pasta frontend -> html.
    Localize o arquivo index.html.
    Dê um clique duplo para abri-lo no seu navegador (Google Chrome, Edge, Firefox, etc).

## 📂 Estrutura de Pastas (Resumo)

DiabetalAI/
├── backend/
│   ├── routes/
│   │   └── router.py  <-- Arquivo principal de execução
│   └── requirements.txt
└── frontend/
    └── html/
        └── index.html <-- Arquivo principal da interface