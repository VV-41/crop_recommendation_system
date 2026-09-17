const chat = document.getElementById("chat");
const form = document.getElementById("predictForm");
const clearBtn = document.getElementById("clearChatBtn");
const quickNote = document.getElementById("quickNote");
const sendNoteBtn = document.getElementById("sendNoteBtn");

function addMsg(text, who="bot", meta="") {
  const wrap = document.createElement("div");
  wrap.className = `msg ${who}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerText = text;

  const m = document.createElement("div");
  m.className = "meta";
  m.innerText = meta || (who === "bot" ? "CropBot" : "You");

  wrap.appendChild(bubble);
  wrap.appendChild(m);
  chat.appendChild(wrap);

  // scroll to bottom
  chat.parentElement.scrollTop = chat.parentElement.scrollHeight;
}

function serializeForm(formEl) {
  const data = {};
  new FormData(formEl).forEach((v, k) => { data[k] = v; });
  return data;
}

function prettySummary(data){
  // Compact single message like WhatsApp
  return `N:${data.Nitrogen}, P:${data.Phosphorus}, K:${data.Potassium} | Rain:${data.Rainfall_mm}mm | Temp:${data.Temperature_C}°C | Hum:${data.Humidity_percent}% | pH:${data.Soil_pH}\nSoil:${data.Soil_Type} | Season:${data.Season} | State:${data.State}\nPrev crop:${data.Previous_Crop}`;
}

form?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = serializeForm(form);

  addMsg(prettySummary(data), "user", "You • sent");

  addMsg("Thinking…", "bot", "CropBot • typing…");
  const typingNode = chat.lastChild;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data),
    });

    const json = await res.json();
    typingNode.remove();

    if (!json.ok) {
      addMsg(`❌ ${json.error}`, "bot", "CropBot");
      return;
    }

    let reply = `✅ Recommended crop: ${json.prediction}`;
    if (json.top_k && json.top_k.length) {
      const lines = json.top_k.map((x, i) => `${i+1}) ${x.crop} — ${(x.probability*100).toFixed(1)}%`);
      reply += `\n\nTop matches:\n${lines.join("\n")}`;
    }
    addMsg(reply, "bot", "CropBot • ready");
  } catch (err) {
    typingNode.remove();
    addMsg("❌ Network error. Is the Flask server running?", "bot", "CropBot");
  }
});

clearBtn?.addEventListener("click", () => {
  // keep the first welcome message
  while (chat.children.length > 1) chat.lastChild.remove();
});

sendNoteBtn?.addEventListener("click", () => {
  const t = (quickNote.value || "").trim();
  if (!t) return;
  addMsg(t, "user", "You • note");
  addMsg("Got it ✅ (Note saved in chat only).", "bot", "CropBot");
  quickNote.value = "";
});