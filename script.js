let currentCharacter = "jarvis";  // default

function setCharacter(name) {
  currentCharacter = name;
  const label = document.getElementById("current-character");
  if (label) {
    label.textContent = name.toUpperCase();
  }
}

async function sendMessage() {
  const input = document.getElementById("chat-input");
  const text = input.value.trim();
  if (!text) return;

  addMessage("You", text);
  input.value = "";

  const resp = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: text,
      character: currentCharacter,
    }),
  });

  const data = await resp.json();
  addMessage(currentCharacter, data.response);
}

// Simple helper to add messages to the chat
function addMessage(sender, text) {
  const box = document.getElementById("chat-box");
  const div = document.createElement("div");
  div.className = "msg";
  div.innerHTML = `<strong>${sender}:</strong> ${text}`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}
