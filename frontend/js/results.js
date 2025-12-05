document.addEventListener("DOMContentLoaded", () => {
    const dadosRaw = localStorage.getItem("resultadoIA");
    if (!dadosRaw) {
        window.location.href = "forms.html";
        return;
    }

    const dados = JSON.parse(dadosRaw);
    const diagnostico = dados.diagnostico;
    const modelos = dados.detalhes_modelos;

    const riscoTexto = document.getElementById("resultado-titulo");
    const riscoDesc = document.getElementById("resultado-descricao");
    const cardDiagnostico = document.getElementById("card-diagnostico");
    const cardCausas = document.getElementById("card-causas");
    const ulFatores = document.getElementById("lista-fatores");
    const areaGraficos = document.getElementById("graficos-area");

    if (diagnostico.tem_diabetes) {
        riscoTexto.innerText = "ALTA PROBABILIDADE";
        riscoTexto.classList.add("risk-positive");
        riscoDesc.innerText = "Os modelos indicam uma tendência para diabetes.";
        
        ulFatores.innerHTML = "";
        diagnostico.principais_causas.forEach(fator => {
            const li = document.createElement("li");
            li.innerHTML = `<strong>${fator}</strong>`;
            ulFatores.appendChild(li);
        });
        
        cardCausas.classList.remove("hidden");
        cardDiagnostico.classList.remove("full-width");
    } else {
        riscoTexto.innerText = "BAIXA PROBABILIDADE";
        riscoTexto.classList.add("risk-negative");
        riscoDesc.innerText = "Os modelos indicam baixo risco no momento.";

        cardCausas.classList.add("hidden");
        cardDiagnostico.classList.add("full-width");
    }

    areaGraficos.innerHTML = "";

    modelos.forEach(modelo => {
        const containerModelo = document.createElement("div");
        containerModelo.className = "model-container";
        
        const titulo = document.createElement("h3");
        titulo.innerText = modelo.tipo_modelo;
        titulo.className = "model-title";
        containerModelo.appendChild(titulo);

        const gridImagens = document.createElement("div");
        gridImagens.className = "charts-row";

        modelo.graficos.forEach((imgBase64, index) => {
            const imgWrapper = document.createElement("div");
            imgWrapper.className = "chart-wrapper";

            const img = document.createElement("img");
            img.src = imgBase64;
            img.alt = `Gráfico ${index + 1} de ${modelo.tipo_modelo}`;
            
            imgWrapper.appendChild(img);
            gridImagens.appendChild(imgWrapper);
        });

        containerModelo.appendChild(gridImagens);
        areaGraficos.appendChild(containerModelo);
    });
});