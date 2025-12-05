document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("form");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const submitBtn = form.querySelector("button[type='submit']");
    const originalText = submitBtn.innerText;

    const Age = Number(form.querySelector('input[placeholder="Ex: 35"]').value);
    const Pregnancies = Number(form.querySelector('input[placeholder="Ex: 2"]').value);
    const BMI = Number(form.querySelector('input[placeholder="Ex: 26.3"]').value);
    const Glucose = Number(form.querySelector('input[placeholder="Ex: 110"]').value);
    const BloodPressure = Number(form.querySelector('input[placeholder="Ex: 80"]').value);
    const Insulin = Number(form.querySelector('input[placeholder="Ex: 90"]').value);
    const DiabetesPedigreeFunction = Number(form.querySelector("select").value);

    const dadosUsuario = {
      Age,
      Pregnancies,
      BMI,
      Glucose,
      BloodPressure,
      Insulin,
      DiabetesPedigreeFunction
    };

    try {
      submitBtn.innerText = "Analisando...";
      submitBtn.disabled = true;

      const resposta = await fetch("http://127.0.0.1:5000/receber-dados", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(dadosUsuario)
      });

      if (!resposta.ok) throw new Error(`Erro HTTP: ${resposta.status}`);

      const dadosRetornados = await resposta.json();
      console.log("Sucesso:", dadosRetornados);

      localStorage.setItem("resultadoIA", JSON.stringify(dadosRetornados.data));

      window.location.href = "results.html";

    } catch (erro) {
      console.error("Erro:", erro);
      alert("Erro ao conectar com a IA. Verifique o backend.");
      submitBtn.innerText = originalText;
      submitBtn.disabled = false;
    }
  });
});