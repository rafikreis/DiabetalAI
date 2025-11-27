document.addEventListener("DOMContentLoaded", () => {
	const dados = JSON.parse(localStorage.getItem("dadosUsuario") || "{}");

	const mockResposta = gerarMockResposta(dados);

	document.getElementById("risco").innerText = mockResposta.probabilidade + "%";
	document.getElementById("nivel-risco").innerText = mockResposta.nivel;

	const ul = document.getElementById("fatores");
	ul.innerHTML = "";
	mockResposta.fatores.forEach(f => {
		const li = document.createElement("li");
		li.innerText = f;
		ul.appendChild(li);
	});

	const bar = document.getElementById("risk-bar");
	setTimeout(() => {
		bar.style.width = mockResposta.probabilidade + "%";
	}, 150);

	desenharGrafico(mockResposta.grafico);

	function gerarMockResposta(dadosUsuario) {

		const base = 8;
		const fromGlicose = (dadosUsuario.glicose || 100) / 200 * 40;
		const fromImc = Math.max(0, ((dadosUsuario.imc || 25) - 20) / 30 * 25);
		const fromIdade = Math.min(15, ((dadosUsuario.idade || 30) - 20) / 60 * 15);
		const sum = base + fromGlicose + fromImc + fromIdade;

		const prob = 100;

		const fatores = [];
		if ((dadosUsuario.glicose || 0) >= 126) fatores.push("Glicose muito alta (>=126 mg/dL)");
		else if ((dadosUsuario.glicose || 0) >= 100) fatores.push("Glicose acima do normal (>=100 mg/dL)");

		if ((dadosUsuario.imc || 0) >= 30) fatores.push("IMC na faixa de obesidade");
		else if ((dadosUsuario.imc || 0) >= 25) fatores.push("IMC acima do recomendado");

		if ((dadosUsuario.historico || 0) > 0.4) fatores.push("Histórico familiar relevante");

		if ((dadosUsuario.pressao || 0) >= 130) fatores.push("Pressão arterial elevada");

		if (fatores.length === 0) {
			fatores.push("Nenhum fator crítico detectado — mantenha hábitos saudáveis");
		}

		const grafico = [
			Math.min(90, Math.round(fromGlicose + 10)),
			Math.min(90, Math.round(fromImc + 20)),
			Math.min(90, Math.round(fromIdade + 15)),
			Math.min(90, prob),
			Math.min(90, Math.round((dadosUsuario.insulina || 50) / 200 * 80))
		];

		const nivel = prob < 20 ? "Baixo" : (prob < 40 ? "Moderado" : (prob < 60 ? "Alto" : "Muito alto"));

		return {
			probabilidade: prob,
			nivel,
			fatores,
			grafico
		};
	}

	function desenharGrafico(valores) {
		const canvas = document.getElementById("grafico");
		const ctx = canvas.getContext("2d");
		ctx.clearRect(0, 0, canvas.width, canvas.height);

		const padding = 40;
		const width = canvas.width - padding * 2;
		const height = canvas.height - padding * 2;

		const barWidth = Math.floor(width / valores.length) - 12;
		valores.forEach((v, i) => {
			const x = padding + i * (barWidth + 12);
			const h = (v / 100) * height;
			const y = canvas.height - padding - h;

			const grad = ctx.createLinearGradient(0, y, 0, y + h);
			grad.addColorStop(0, "#2bb7ff");
			grad.addColorStop(1, "#ffb86b");

			ctx.fillStyle = grad;
			ctx.fillRect(x, y, barWidth, h);

			ctx.fillStyle = "#274f48";
			ctx.font = "13px Poppins";
			ctx.textAlign = "center";
			ctx.fillText(v + "%", x + barWidth / 2, y - 8);
		});
	}
});
