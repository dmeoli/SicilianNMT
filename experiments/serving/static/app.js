// Minimal front-end: calls POST /translate on our FastAPI NLLB service.
const LANGS = { scn: "Sicilianu", en: "English" };
let dir = { src: "scn", tgt: "en" };

const $ = (id) => document.getElementById(id);
const srcBox = $("src"), tgtBox = $("tgt"), statusEl = $("status"), goBtn = $("go");

function renderDir() {
  $("srcLabel").textContent = LANGS[dir.src];
  $("tgtLabel").textContent = LANGS[dir.tgt];
  srcBox.placeholder = dir.src === "scn" ? "Scrivi cc…" : "Type here…";
}

$("swap").addEventListener("click", () => {
  dir = { src: dir.tgt, tgt: dir.src };
  // swap any existing text too, so a round-trip is easy
  const s = srcBox.value; srcBox.value = tgtBox.value; tgtBox.value = s;
  renderDir();
  srcBox.focus();
});

async function translate() {
  const text = srcBox.value.trim();
  if (!text) { srcBox.focus(); return; }
  goBtn.disabled = true;
  statusEl.textContent = "…";
  tgtBox.value = "";
  try {
    const r = await fetch("/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, src: dir.src, tgt: dir.tgt }),
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    tgtBox.value = data.translation || "";
    statusEl.textContent = "";
  } catch (e) {
    statusEl.textContent = "Erruri / error: " + e.message;
  } finally {
    goBtn.disabled = false;
  }
}

goBtn.addEventListener("click", translate);
// Ctrl/Cmd + Enter to translate
srcBox.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") translate();
});

renderDir();
