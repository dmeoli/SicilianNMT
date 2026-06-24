// Front-end for the trilingual (scn/en/it) translator. Calls POST /translate.
const $ = (id) => document.getElementById(id);
const srcBox = $("src"), tgtBox = $("tgt"), statusEl = $("status"), goBtn = $("go");
const srcLang = $("srcLang"), tgtLang = $("tgtLang");

// keep source and target languages distinct
function ensureDistinct(changed) {
  if (srcLang.value === tgtLang.value) {
    const other = changed === srcLang ? tgtLang : srcLang;
    other.value = ["scn", "en", "it"].find((l) => l !== changed.value);
  }
}
srcLang.addEventListener("change", () => ensureDistinct(srcLang));
tgtLang.addEventListener("change", () => ensureDistinct(tgtLang));

$("swap").addEventListener("click", () => {
  const s = srcLang.value; srcLang.value = tgtLang.value; tgtLang.value = s;
  const t = srcBox.value; srcBox.value = tgtBox.value; tgtBox.value = t;
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
      body: JSON.stringify({ text, src: srcLang.value, tgt: tgtLang.value }),
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
srcBox.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") translate();
});
