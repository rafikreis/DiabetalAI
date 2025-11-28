document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("form");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const idade = Number(form.querySelector('input[placeholder="Ex: 35"]').value);
    const gestacoes = Number(form.querySelector('input[placeholder="Ex: 2"]').value);
    const imc = Number(form.querySelector('input[placeholder="Ex: 26.3"]').value);

    const glicose = Number(form.querySelector('input[placeholder="Ex: 110"]').value);
    const pressao = Number(form.querySelector('input[placeholder="Ex: 80"]').value);
    const insulina = Number(form.querySelector('input[placeholder="Ex: 90"]').value);

    const historico = Number(form.querySelector("select").value);

    const dadosUsuario = {
      idade,
      gestacoes,
      imc,
      glicose,
      pressao,
      insulina,
      historico
    };

    try {
      const resposta = await fetch("http://127.0.0.1:5000/receber-dados", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(dadosUsuario)
      });

      const resultado = await resposta.json();
      console.log("Servidor respondeu:", resultado);

      localStorage.setItem("dadosUsuario", JSON.stringify(dadosUsuario));

      window.location.href = "results.html";  

    } catch (erro) {
      console.error("Erro ao enviar:", erro);
    }
  });
});