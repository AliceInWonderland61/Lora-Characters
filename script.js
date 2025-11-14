// ----------------------------
// Character switching
// ----------------------------

let currentCharacter = "jarvis";  // default character

function setCharacter(name) {
    currentCharacter = name;

    // Update the UI label
    const label = document.getElementById("current-character");
    if (label) {
        label.textContent = name.toUpperCase();
    }
}

// ----------------------------
// Add messages to chat box
// ----------------------------

function addMessage(sender, text) {
    const box = document.getElementById("chat-box");

    const msg = document.createElement("div");
    msg.className = "message";
    msg.innerHTML = `<strong>${sender}:</strong> ${text}`;

    box.appendChild(msg);
    box.scrollTop = box.scrollHeight;
}

// ----------------------------
// Send user input to backend
// ----------------------------

async function sendMessage() {
    const input = document.getElementById("chat-input");
    const text = input.value.trim();

    if (!text) return;

    // Show user message
    addMessage("You", text);
    input.value = "";

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: text,
                character: currentCharacter,
            }),
        });

        const data = await response.json();

        // Show model reply
        addMessage(currentCharacter, data.response);

    } catch (err) {
        addMessage("System", "⚠️ Error contacting server.");
        console.error(err);
    }
}
