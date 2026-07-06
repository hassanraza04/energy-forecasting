const state = {
  config: null,
  values: {},
};

const formatValue = (value, unit) => {
  const formatted = Number(value).toLocaleString(undefined, {
    maximumFractionDigits: 1,
  });
  return unit ? `${formatted} ${unit}` : formatted;
};

const setText = (id, value) => {
  document.getElementById(id).textContent = value;
};

async function loadConfig() {
  const response = await fetch("/api/config");
  state.config = await response.json();
  state.config.inputs.forEach((input) => {
    state.values[input.key] = input.default;
  });
  renderConfig();
  await runPrediction();
}

function renderConfig() {
  setText("modelName", state.config.model.name);
  setText("averageValue", `${state.config.averageEnergy} Wh`);
  setText("maeValue", `${state.config.model.mae} Wh`);
  setText(
    "modelNotes",
    `${state.config.model.name} uses saved parameters ${JSON.stringify(state.config.model.params)}. R2 is ${state.config.model.r2}, measured on the held out test split.`,
  );

  const presets = document.getElementById("presets");
  presets.innerHTML = "";
  state.config.presets.forEach((preset) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset";
    button.textContent = preset.name;
    button.addEventListener("click", () => applyPreset(preset.values));
    presets.appendChild(button);
  });

  const inputs = document.getElementById("inputs");
  inputs.innerHTML = "";
  state.config.inputs.forEach((input) => {
    const field = document.createElement("label");
    field.className = "field";
    field.innerHTML = `
      <span class="field-header">
        <span>${input.label}</span>
        <output id="out-${input.key}">${formatValue(state.values[input.key], input.unit)}</output>
      </span>
      <input
        id="input-${input.key}"
        type="range"
        min="${input.min}"
        max="${input.max}"
        step="${input.step}"
        value="${state.values[input.key]}"
      >
    `;
    inputs.appendChild(field);
    field.querySelector("input").addEventListener("input", (event) => {
      state.values[input.key] = Number(event.target.value);
      document.getElementById(`out-${input.key}`).textContent = formatValue(
        state.values[input.key],
        input.unit,
      );
    });
  });
}

function applyPreset(values) {
  Object.assign(state.values, values);
  state.config.inputs.forEach((input) => {
    const control = document.getElementById(`input-${input.key}`);
    const output = document.getElementById(`out-${input.key}`);
    control.value = state.values[input.key];
    output.textContent = formatValue(state.values[input.key], input.unit);
  });
  runPrediction();
}

async function runPrediction() {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.values),
  });
  const result = await response.json();
  setText("predictionValue", `${result.prediction} ${result.unit}`);
  setText("heroPrediction", Math.round(result.prediction));
  setText(
    "predictionText",
    `${result.description.message} It is ${result.deltaPercent}% against the saved dataset average.`,
  );
}

document.getElementById("forecastForm").addEventListener("submit", (event) => {
  event.preventDefault();
  runPrediction();
});

loadConfig();
