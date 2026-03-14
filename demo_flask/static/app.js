// Reset session on new browser tab
(() => {
  const params = new URLSearchParams(window.location.search);
  const initialized = sessionStorage.getItem("sessionInit");
  if (!initialized && !params.has("reset")) {
    sessionStorage.setItem("sessionInit", "1");
    window.location.replace("/?reset=1");
  } else if (!initialized) {
    sessionStorage.setItem("sessionInit", "1");
  }
})();

// Restore scroll position or scroll to bottom for new messages
const chatWindow = document.getElementById("chatWindow");
if (chatWindow) {
  const savedScrollPos = sessionStorage.getItem("chatScrollPos");
  if (savedScrollPos !== null) {
    requestAnimationFrame(() => {
      chatWindow.scrollTop = parseInt(savedScrollPos, 10);
      sessionStorage.removeItem("chatScrollPos");
    });
  } else {
    requestAnimationFrame(() => {
      chatWindow.scrollTop = chatWindow.scrollHeight;
    });
  }
}

// Save scroll position before form submission
function saveScrollPosition() {
  if (chatWindow) {
    sessionStorage.setItem("chatScrollPos", chatWindow.scrollTop.toString());
  }
}

// Attach to all forms that cause page reload
document.querySelectorAll("form").forEach((form) => {
  form.addEventListener("submit", saveScrollPosition);
});

// Speech Recognition handling
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let activeRecognition = null;
let activeRole = null;

function setStatus(role, text, isActive) {
  const status = document.getElementById(`${role}Status`);
  const button = document.getElementById(`${role}MicBtn`);
  const micText = button ? button.querySelector(".mic-text") : null;
  if (status) status.textContent = text;
  if (button) button.classList.toggle("active", Boolean(isActive));
  if (micText) micText.textContent = isActive ? "Stop" : "Start Mic";
}

function setupMic(role) {
  const button = document.getElementById(`${role}MicBtn`);
  const transcript = document.getElementById(`${role}Transcript`);
  const input = document.getElementById(`${role}Input`);
  const sourceInput = document.getElementById(`${role}Source`);

  if (!SpeechRecognition) {
    setStatus(role, "Not supported", false);
    if (button) button.disabled = true;
    if (transcript) transcript.textContent = "Voice not supported. Use Chrome or Edge.";
    return;
  }

  // Check if running on HTTPS or localhost (required for Web Speech API)
  const isSecureContext = window.isSecureContext || location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
  if (!isSecureContext) {
    setStatus(role, "Requires HTTPS", false);
    if (button) button.disabled = true;
    if (transcript) transcript.textContent = "Voice requires HTTPS. Use localhost or deploy with HTTPS.";
    return;
  }

  button.addEventListener("click", () => {
    if (activeRecognition && activeRole === role) {
      activeRecognition.stop();
      return;
    }
    if (activeRecognition) {
      activeRecognition.stop();
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = true;

    let finalText = input.value ? `${input.value} ` : "";

    recognition.onstart = () => {
      activeRecognition = recognition;
      activeRole = role;
      setStatus(role, "Listening...", true);
      if (transcript) {
        transcript.textContent = "Listening... speak now.";
        transcript.classList.add("listening");
      }
      if (sourceInput) sourceInput.value = "voice";
    };

    recognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          finalText += result[0].transcript.trim() + " ";
        } else {
          interim += result[0].transcript;
        }
      }
      const displayText = `${finalText}${interim}`.trim();
      if (transcript) transcript.textContent = displayText || "Listening...";
      input.value = finalText.trim();
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
      let msg = "Error";
      if (event.error === "not-allowed") {
        msg = "Mic blocked - please allow microphone access";
      } else if (event.error === "no-speech") {
        msg = "No speech detected";
      } else if (event.error === "network") {
        msg = "Network error";
      } else if (event.error === "audio-capture") {
        msg = "No microphone found";
      }
      setStatus(role, msg, false);
      if (transcript) {
        transcript.classList.remove("listening");
        transcript.textContent = msg;
      }
    };

    recognition.onend = () => {
      setStatus(role, "Ready", false);
      if (activeRecognition === recognition) {
        activeRecognition = null;
        activeRole = null;
      }
      if (transcript) {
        transcript.classList.remove("listening");
        if (!input.value) transcript.textContent = "Voice transcript will appear here...";
      }
      // Don't auto-send - let user review and click Send button manually
    };

    try {
      recognition.start();
    } catch (err) {
      console.error("Failed to start speech recognition:", err);
      setStatus(role, "Failed to start", false);
      if (transcript) transcript.textContent = "Failed to start microphone. Please try again.";
    }
  });

  input.addEventListener("input", () => {
    if (sourceInput) sourceInput.value = "text";
  });
}

setupMic("researcher");
setupMic("expert");

// Custom file input - auto-submit on file selection
const fileInput = document.getElementById("guideFile");
const fileLabel = document.querySelector(".file-text");
if (fileInput && fileLabel) {
  const uploadForm = fileInput.closest("form");
  fileInput.addEventListener("change", () => {
    const selectedFile = fileInput.files[0];
    fileLabel.textContent = fileInput.files[0]?.name || "Upload interview script (txt, docx, pdf)";
    if (selectedFile && uploadForm) {
      saveScrollPosition();
      uploadForm.submit();
    }
  });
}

// ============== Resizable Two-Panel Layout ==============

function initPanelResizers() {
  const container = document.querySelector(".main-panels");
  if (!container) return;

  const leftPanel = container.querySelector(".script-panel");
  const leftResizer = container.querySelector('.panel-resizer[data-resizer="left"]');
  if (!leftPanel || !leftResizer) return;

  const minLeft = 240;
  const minCenter = 420;

  let isDragging = false;
  let startX = 0;
  let startLeftWidth = 0;

  const stopDragging = () => {
    if (!isDragging) return;
    isDragging = false;
    leftResizer.classList.remove("is-dragging");
    document.body.classList.remove("is-resizing-panels");
    window.removeEventListener("mousemove", onDragMove);
    window.removeEventListener("mouseup", stopDragging);
  };

  const onDragMove = (event) => {
    if (!isDragging) return;
    const dx = event.clientX - startX;
    const totalWidth = container.clientWidth;
    let nextLeft = startLeftWidth + dx;
    const maxLeft = totalWidth - minCenter - 8;
    nextLeft = Math.max(minLeft, Math.min(maxLeft, nextLeft));
    container.style.setProperty("--left-panel-width", `${nextLeft}px`);
  };

  leftResizer.addEventListener("mousedown", (event) => {
    if (window.innerWidth <= 1024) return;
    isDragging = true;
    startX = event.clientX;
    startLeftWidth = leftPanel.getBoundingClientRect().width;
    document.body.classList.add("is-resizing-panels");
    leftResizer.classList.add("is-dragging");
    window.addEventListener("mousemove", onDragMove);
    window.addEventListener("mouseup", stopDragging);
    event.preventDefault();
  });

  window.addEventListener("blur", stopDragging);
}

initPanelResizers();
