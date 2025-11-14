// Create falling leaves
function createLeaf() {
    const leaf = document.createElement("img");
    leaf.src = "https://i.imgur.com/1Q9Z1EM.png"; // simple leaf PNG
    leaf.classList.add("leaf");

    leaf.style.left = Math.random() * 100 + "vw";
    leaf.style.animationDuration = 4 + Math.random() * 6 + "s";
    leaf.style.opacity = 0.6 + Math.random() * 0.4;

    document.body.appendChild(leaf);

    setTimeout(() => {
        leaf.remove();
    }, 10000);
}

setInterval(createLeaf, 600);
