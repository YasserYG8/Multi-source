// Test page for the sentiment API. Plain JavaScript, no build step.
"use strict";

const API = ""; // same origin: the page is served by FastAPI

const LABELS = { positif: "Positif", neutre: "Neutre", negatif: "Négatif" };
const SOURCE_LABELS = { texte: "Texte", image: "Image", audio: "Audio" };
const CONVERSION = { texte: null, image: "OCR (Tesseract)", audio: "Transcription (Whisper)" };

// New sentences, not taken from the training set.
const TEXT_EXAMPLES = [
  "Livraison en avance et produit encore mieux que sur les photos, merci !",
  "Le colis est arrivé mardi, conforme à la commande.",
  "Écran rayé à la réception et aucune réponse du vendeur depuis dix jours.",
];

const $ = (selector) => document.querySelector(selector);

const state = {
  source: "texte",
  files: { image: null, audio: null },   // { blob, name, ref }
  busy: false,
  history: [],
  recorder: null,
};

// ---------- Helpers ----------

const formatNumber = (value, digits = 2) =>
  value.toLocaleString("fr-FR", { minimumFractionDigits: digits, maximumFractionDigits: digits });

const formatDuration = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${formatNumber(ms / 1000, 1)} s`);

function icon(name, extraClass = "") {
  return `<svg class="icon ${extraClass}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ---------- API status ----------

async function checkHealth() {
  const status = $("#api-status");
  try {
    const response = await fetch(`${API}/health`);
    if (!response.ok) throw new Error();
    status.dataset.state = "online";
    status.querySelector(".status__label").textContent = "API en ligne";
  } catch {
    status.dataset.state = "offline";
    status.querySelector(".status__label").textContent = "API hors ligne";
  }
}

// ---------- Tabs ----------

function selectSource(source, { focus = false } = {}) {
  state.source = source;
  document.querySelectorAll('[role="tab"]').forEach((tab) => {
    const selected = tab.dataset.source === source;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    if (selected && focus) tab.focus();
  });
  document.querySelectorAll('[role="tabpanel"]').forEach((panel) => {
    panel.hidden = panel.id !== `panel-${source}`;
  });
  updateSubmit();
}

function setupTabs() {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectSource(tab.dataset.source));
    tab.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      const next = tabs[(index + step + tabs.length) % tabs.length];
      selectSource(next.dataset.source, { focus: true });
    });
  });
}

// ---------- Text ----------

function setupText() {
  const textarea = $("#texte");
  const counter = $("#counter");
  const update = () => {
    counter.textContent = `${textarea.value.length} / 5000`;
    updateSubmit();
  };
  textarea.addEventListener("input", update);
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      $("#form").requestSubmit();
    }
  });

  const chips = $("#text-examples");
  TEXT_EXAMPLES.forEach((sentence) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = sentence;
    chip.addEventListener("click", () => {
      textarea.value = sentence;
      update();
      textarea.focus();
    });
    chips.append(chip);
  });
  update();
}

// ---------- Files (image / audio) ----------

function setFile(kind, blob, name, ref = null) {
  state.files[kind] = blob ? { blob, name, ref } : null;
  const drop = $(`#drop-${kind}`);
  const preview = drop.querySelector(".drop__preview");
  const media = $(`#preview-${kind}`);

  if (media.dataset.url) URL.revokeObjectURL(media.dataset.url);
  if (blob) {
    const url = URL.createObjectURL(blob);
    media.src = url;
    media.dataset.url = url;
    preview.querySelector(".file-name").textContent = name;
  } else {
    media.removeAttribute("src");
    delete media.dataset.url;
    $(`#file-${kind}`).value = "";
  }
  preview.hidden = !blob;
  drop.classList.toggle("has-file", Boolean(blob));

  // Mark the matching dataset example as pressed.
  document.querySelectorAll(`[data-example-kind="${kind}"]`).forEach((button) => {
    button.setAttribute("aria-pressed", String(Boolean(blob) && button.dataset.name === name));
  });
  updateSubmit();
}

function setupDrop(kind) {
  const drop = $(`#drop-${kind}`);
  const input = $(`#file-${kind}`);

  input.addEventListener("change", () => {
    const file = input.files[0];
    if (file) setFile(kind, file, file.name);
  });
  drop.querySelector(`[data-clear="${kind}"]`).addEventListener("click", () => setFile(kind, null));

  ["dragenter", "dragover"].forEach((type) =>
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.add("is-over");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    drop.addEventListener(type, () => drop.classList.remove("is-over"))
  );
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) setFile(kind, file, file.name);
  });
}

// Paste a screenshot anywhere on the page (Ctrl + V).
function setupPaste() {
  document.addEventListener("paste", (event) => {
    const item = [...(event.clipboardData?.items || [])].find((entry) => entry.type.startsWith("image/"));
    if (!item) return;
    event.preventDefault();
    const blob = item.getAsFile();
    selectSource("image");
    setFile("image", blob, "capture-collee.png");
  });
}

// ---------- Microphone recording ----------

function setupRecorder() {
  const button = $("#record");
  const time = $("#record-time");
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    button.disabled = true;
    button.querySelector("span").textContent = "Micro non disponible";
    return;
  }

  let timer = null;
  const stop = () => state.recorder?.state === "recording" && state.recorder.stop();

  button.addEventListener("click", async () => {
    if (state.recorder?.state === "recording") return stop();
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      showError("Micro inaccessible", ["Autorisez l'accès au micro dans le navigateur, puis réessayez."]);
      return;
    }
    const chunks = [];
    const recorder = new MediaRecorder(stream);
    state.recorder = recorder;
    recorder.addEventListener("dataavailable", (event) => event.data.size && chunks.push(event.data));
    recorder.addEventListener("stop", () => {
      clearInterval(timer);
      stream.getTracks().forEach((track) => track.stop());
      button.classList.remove("is-recording");
      button.querySelector("span").textContent = "Enregistrer au micro";
      button.querySelector("use").setAttribute("href", "#i-record");
      time.textContent = "";
      const type = recorder.mimeType || "audio/webm";
      const extension = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm";
      setFile("audio", new Blob(chunks, { type }), `enregistrement.${extension}`);
    });

    const started = Date.now();
    recorder.start();
    button.classList.add("is-recording");
    button.querySelector("span").textContent = "Arrêter l'enregistrement";
    button.querySelector("use").setAttribute("href", "#i-stop");
    time.textContent = "0:00";
    timer = setInterval(() => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      time.textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
      if (seconds >= 60) stop(); // keep files small
    }, 250);
  });
}

// ---------- Dataset examples ----------

async function loadExamples() {
  let data;
  try {
    const response = await fetch(`${API}/demo/exemples`);
    if (!response.ok) return;
    data = await response.json();
  } catch {
    return;
  }

  const pick = async (kind, example) => {
    const response = await fetch(example.url);
    const blob = await response.blob();
    setFile(kind, blob, example.fichier, example.sentiment_ref);
  };

  if (data.images.length) {
    const grid = $("#image-examples");
    data.images.forEach((example) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "thumb";
      button.dataset.exampleKind = "image";
      button.dataset.name = example.fichier;
      button.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-label", `${example.fichier}, sentiment attendu : ${LABELS[example.sentiment_ref]}`);
      button.innerHTML = `<img src="${example.url}" alt="" loading="lazy"><span><i class="dot" data-sentiment="${example.sentiment_ref}"></i>${escapeHtml(example.fichier)}</span>`;
      button.addEventListener("click", () => pick("image", example));
      grid.append(button);
    });
    $("#image-examples-block").hidden = false;
  }

  if (data.audios.length) {
    const chips = $("#audio-examples");
    data.audios.forEach((example) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "chip";
      button.dataset.exampleKind = "audio";
      button.dataset.name = example.fichier;
      button.setAttribute("aria-pressed", "false");
      button.title = example.texte_ref;
      button.setAttribute("aria-label", `${example.fichier}, sentiment attendu : ${LABELS[example.sentiment_ref]}`);
      button.innerHTML = `<i class="dot" data-sentiment="${example.sentiment_ref}"></i>${escapeHtml(example.fichier)}`;
      button.addEventListener("click", () => pick("audio", example));
      chips.append(button);
    });
    $("#audio-examples-block").hidden = false;
  }
}

// ---------- Submit ----------

function isReady() {
  if (state.source === "texte") return $("#texte").value.trim().length > 0;
  return Boolean(state.files[state.source]);
}

function updateSubmit() {
  const button = $("#submit");
  button.disabled = state.busy || !isReady();
  button.querySelector("span").textContent = state.busy ? "Analyse en cours…" : "Analyser l'avis";
}

function buildRequest() {
  if (state.source === "texte") {
    return {
      url: `${API}/predict/text`,
      options: {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texte: $("#texte").value }),
      },
      name: null,
      ref: null,
    };
  }
  const { blob, name, ref } = state.files[state.source];
  const form = new FormData();
  form.append("fichier", blob, name);
  return { url: `${API}/predict/${state.source}`, options: { method: "POST", body: form }, name, ref };
}

async function submit(event) {
  event.preventDefault();
  if (state.busy || !isReady()) return;

  const source = state.source;
  const request = buildRequest();
  state.busy = true;
  updateSubmit();
  showLoading(source, request.name);

  const started = performance.now();
  try {
    const response = await fetch(request.url, request.options);
    const body = await response.json().catch(() => null);
    const duration = performance.now() - started;
    if (!response.ok) {
      const messages = Array.isArray(body?.detail)
        ? body.detail.map((item) => item.msg)
        : [body?.detail || `Erreur HTTP ${response.status}`];
      showRefused(source, messages, response.status);
      addHistory({ source, name: request.name, text: messages.join(" "), sentiment: "refuse", duration });
      return;
    }
    showResult(body, { name: request.name, ref: request.ref, duration });
    addHistory({ source, name: request.name, text: body.texte_extrait, sentiment: body.sentiment_predit,
                 confidence: body.confiance, ref: request.ref, duration });
  } catch {
    showRefused(source, ["L'API ne répond pas. Lancez-la avec : uvicorn main:app --reload"], null);
    $("#api-status").dataset.state = "offline";
    $("#api-status .status__label").textContent = "API hors ligne";
  } finally {
    state.busy = false;
    updateSubmit();
  }
}

// ---------- Rendering the label ----------

function setRoute(source, stage) {
  // stage: "loading" | "done" | "refused"
  const stops = Object.fromEntries([...document.querySelectorAll(".route__stop")].map((stop) => [stop.dataset.stop, stop]));
  $("#route-source").textContent = SOURCE_LABELS[source];
  stops.source.dataset.state = "done";

  const conversion = CONVERSION[source];
  $("#route-convert").textContent = conversion || "Aucune (déjà du texte)";
  stops.convert.dataset.state = conversion ? (stage === "loading" ? "active" : "done") : "skipped";
  stops.model.dataset.state = stage === "loading" ? (conversion ? "idle" : "active") : stage === "refused" ? "skipped" : "done";
  stops.result.dataset.state = stage === "done" ? "done" : "idle";
}

function setStamp(kind, text) {
  const stamp = $("#stamp");
  stamp.dataset.sentiment = kind;
  stamp.textContent = text;
  stamp.classList.remove("is-landing");
  if (["positif", "neutre", "negatif", "refuse"].includes(kind)) {
    void stamp.offsetWidth; // restart the animation
    stamp.classList.add("is-landing");
  }
}

function setGauge(value, sentiment) {
  const gauge = $("#gauge");
  gauge.dataset.sentiment = sentiment || "";
  $("#gauge-fill").style.width = `${Math.round((value || 0) * 100)}%`;
  if (value == null) {
    $("#conf-value").textContent = "—";
    gauge.setAttribute("aria-label", "Pas de résultat");
  } else {
    $("#conf-value").innerHTML = `${formatNumber(value)}<small>${Math.round(value * 100)} %</small>`;
    gauge.setAttribute("aria-label", `Confiance ${formatNumber(value)} sur 1`);
  }
}

function setExtracted(source, html) {
  $("#contents-label").textContent = source === "texte" ? "Texte analysé" : source === "image" ? "Texte lu par l'OCR" : "Transcription";
  $("#extracted").innerHTML = html;
}

function showLoading(source, name) {
  $("#label").setAttribute("aria-busy", "true");
  $("#error").hidden = true;
  $("#check").hidden = true;
  setRoute(source, "loading");
  $("#route-result").textContent = "—";
  setStamp("attente", source === "audio" ? "Écoute…" : source === "image" ? "Lecture…" : "Tri…");
  setGauge(null);
  const wait = source === "audio" ? "La transcription prend quelques secondes." : source === "image" ? "Lecture de l'image par l'OCR…" : "Classification du texte…";
  setExtracted(source, `<p class="placeholder">${wait}</p>`);
  $("#meta").textContent = name || "";
}

function showResult(result, { name, ref, duration }) {
  $("#label").setAttribute("aria-busy", "false");
  setRoute(result.source, "done");
  $("#route-result").textContent = LABELS[result.sentiment_predit];
  setStamp(result.sentiment_predit, LABELS[result.sentiment_predit]);
  setGauge(result.confiance, result.sentiment_predit);
  setExtracted(result.source, `<p>${escapeHtml(result.texte_extrait)}</p>`);
  $("#meta").textContent = [name, `analysé en ${formatDuration(duration)}`].filter(Boolean).join(" · ");

  const check = $("#check");
  if (ref) {
    const ok = ref === result.sentiment_predit;
    check.dataset.ok = String(ok);
    check.innerHTML = `${icon(ok ? "check" : "cross")}Attendu dans le dataset : ${LABELS[ref]}${ok ? "" : " (erreur du modèle)"}`;
    check.hidden = false;
  } else {
    check.hidden = true;
  }
}

function showRefused(source, messages, status) {
  $("#label").setAttribute("aria-busy", "false");
  setRoute(source, "refused");
  $("#route-result").textContent = status ? `Erreur ${status}` : "Pas de réponse";
  setStamp("refuse", "Refusé");
  setGauge(null);
  setExtracted(source, `<p class="placeholder">Aucun texte analysé.</p>`);
  $("#meta").textContent = "";
  showError(status === 422 ? "Données refusées par l'API (422)" : "Analyse impossible", messages);
}

function showError(title, messages) {
  const box = $("#error");
  box.innerHTML = `<strong>${escapeHtml(title)}</strong><ul>${messages.map((m) => `<li>${escapeHtml(m)}</li>`).join("")}</ul>`;
  box.hidden = false;
}

// ---------- History ----------

function addHistory(entry) {
  state.history.unshift({ ...entry, index: state.history.length + 1 });
  const rows = state.history.map((item) => {
    const sentiment = item.sentiment === "refuse"
      ? `<span class="tag" data-sentiment="refuse">Refusé</span>`
      : `<span class="tag" data-sentiment="${item.sentiment}"><i class="dot" data-sentiment="${item.sentiment}"></i>${LABELS[item.sentiment]}</span>`;
    const expected = item.ref
      ? `<span class="expected" data-ok="${item.ref === item.sentiment}">${icon(item.ref === item.sentiment ? "check" : "cross")}${LABELS[item.ref]}</span>`
      : `<span class="muted">—</span>`;
    const source = `${SOURCE_LABELS[item.source]}${item.name ? `<br><span class="muted">${escapeHtml(item.name)}</span>` : ""}`;
    return `<tr>
      <td>${item.index}</td>
      <td>${source}</td>
      <td class="excerpt"><span>${escapeHtml(item.text)}</span></td>
      <td>${sentiment}</td>
      <td class="num">${item.confidence != null ? formatNumber(item.confidence) : "—"}</td>
      <td class="tag-cell">${expected}</td>
      <td class="num">${formatDuration(item.duration)}</td>
    </tr>`;
  });
  $("#ledger-rows").innerHTML = rows.join("");

  const withRef = state.history.filter((item) => item.ref && item.sentiment !== "refuse");
  const correct = withRef.filter((item) => item.ref === item.sentiment).length;
  const count = state.history.length;
  $("#ledger-summary").textContent =
    `${count} envoi${count > 1 ? "s" : ""}` +
    (withRef.length ? ` · ${correct} / ${withRef.length} corrects sur les exemples du dataset` : "");
}

// ---------- Start ----------

setupTabs();
setupText();
setupDrop("image");
setupDrop("audio");
setupPaste();
setupRecorder();
$("#form").addEventListener("submit", submit);
checkHealth();
loadExamples();
