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
    // Use requestAnimationFrame to ensure DOM is fully rendered before restoring scroll
    requestAnimationFrame(() => {
      chatWindow.scrollTop = parseInt(savedScrollPos, 10);
      sessionStorage.removeItem("chatScrollPos");
    });
  } else {
    // First load - scroll to bottom
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

// Context slider behavior
document.querySelectorAll(".context-slider").forEach((block) => {
  const slider = block.querySelector('input[type="range"]');
  const output = block.querySelector(".context-output");
  const mapping = [
    block.dataset.coarse || "",
    block.dataset.balanced || "",
    block.dataset.fine || "",
  ];
  if (slider && output) {
    slider.addEventListener("input", (event) => {
      output.textContent = mapping[Number(event.target.value)] || "";
    });
  }
});

// ============== Inline Jargon Highlighting ==============
// Highlight jargon terms directly in expert message text with hover tooltips

function highlightJargonInText(element) {
  const jargonData = element.dataset.jargon;
  if (!jargonData) return;
  
  let jargonList;
  try {
    jargonList = JSON.parse(jargonData);
  } catch (e) {
    return;
  }
  
  if (!jargonList || jargonList.length === 0) return;
  
  let html = element.textContent;
  
  // Sort by term length (longest first) to avoid partial replacements
  jargonList.sort((a, b) => b.term.length - a.term.length);
  
  // Replace each jargon term with highlighted version
  jargonList.forEach((item) => {
    const term = item.term;
    const desc = item.desc || "";
    // Escape special regex characters in term
    const escapedTerm = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    // Create case-insensitive regex for whole word match
    const regex = new RegExp(`\\b(${escapedTerm})\\b`, "gi");
    html = html.replace(regex, `<span class="inline-jargon" data-tooltip="${desc.replace(/"/g, '&quot;')}">$1</span>`);
  });
  
  element.innerHTML = html;
}

// Apply jargon highlighting to all expert messages
document.querySelectorAll(".expert-content").forEach(highlightJargonInText);

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
  const form = document.getElementById(`${role}Form`);

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
    // Use English for speech recognition
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
      console.log("Speech recognition started with language:", recognition.lang);
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

// Custom file input label
const fileInput = document.getElementById("guideFile");
const fileLabel = document.querySelector(".file-text");
if (fileInput && fileLabel) {
  fileInput.addEventListener("change", () => {
    fileLabel.textContent = fileInput.files[0]?.name || "Choose file (txt, docx, pdf)";
  });
}

// Selection tooltip for expert domain concepts
const selectionTooltip = document.getElementById("selectionTooltip");

function hideSelectionTooltip() {
  if (selectionTooltip) selectionTooltip.classList.add("hidden");
}

function showSelectionTooltip(rect, term, desc, found) {
  if (!selectionTooltip) return;
  selectionTooltip.innerHTML = `
    <div class="selection-title">Selected term</div>
    <div class="selection-term">${term}</div>
    <div class="selection-desc">${desc}</div>
  `;
  selectionTooltip.style.top = `${Math.max(10, rect.top - 10)}px`;
  selectionTooltip.style.left = `${Math.min(window.innerWidth - 340, rect.left)}px`;
  selectionTooltip.classList.remove("hidden");
}

document.addEventListener("mouseup", (event) => {
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed) {
    hideSelectionTooltip();
    return;
  }
  const selectedText = selection.toString().trim();
  if (!selectedText) {
    hideSelectionTooltip();
    return;
  }

  const anchorNode = selection.anchorNode;
  const contentEl = anchorNode ? anchorNode.parentElement?.closest(".expert-content") : null;
  if (!contentEl) {
    hideSelectionTooltip();
    return;
  }

  let jargonList = [];
  try {
    jargonList = JSON.parse(contentEl.dataset.jargon || "[]");
  } catch (err) {
    jargonList = [];
  }

  const match = jargonList.find((item) => {
    const term = String(item.term || "").toLowerCase();
    const selected = selectedText.toLowerCase();
    return term && (selected === term || selected.includes(term) || term.includes(selected));
  });

  const rect = selection.getRangeAt(0).getBoundingClientRect();
  const description = match
    ? match.desc
    : "No detected domain concept for this selection. Consider rephrasing or tagging manually.";

  showSelectionTooltip(rect, selectedText, description, Boolean(match));
});

document.addEventListener("click", (event) => {
  if (!selectionTooltip) return;
  if (!selectionTooltip.contains(event.target)) {
    hideSelectionTooltip();
  }
});

// ============== Action Buttons for On-Demand Assistance ==============

/**
 * Generic handler for action buttons that call backend API
 * @param {string} endpoint - API endpoint (e.g., '/api/get_refinement')
 * @param {number} msgIndex - Message index
 * @param {HTMLButtonElement} button - The button element
 */
async function callAssistanceAPI(endpoint, msgIndex, button) {
  const originalText = button.querySelector(".action-text")?.textContent || "Loading...";
  const actionText = button.querySelector(".action-text");
  
  // Show loading state
  button.disabled = true;
  button.classList.add("loading");
  if (actionText) actionText.textContent = "Generating...";
  
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ msg_index: msgIndex }),
    });
    
    const result = await response.json();
    
    if (result.success) {
      // Save scroll position and reload page to show the persisted result
      saveScrollPosition();
      window.location.reload();
    } else {
      // Show error and reset button
      button.disabled = false;
      button.classList.remove("loading");
      if (actionText) actionText.textContent = originalText;
      alert(`Error: ${result.error || "Failed to generate assistance"}`);
    }
  } catch (error) {
    // Network error - reset button
    button.disabled = false;
    button.classList.remove("loading");
    if (actionText) actionText.textContent = originalText;
    alert(`Network error: ${error.message}`);
  }
}

/**
 * Handle "Get Refinement" button click
 */
function handleRefinementClick(event) {
  const button = event.currentTarget;
  const msgIndex = parseInt(button.dataset.msgIndex, 10);
  if (isNaN(msgIndex)) return;
  callAssistanceAPI("/api/get_refinement", msgIndex, button);
}

/**
 * Handle "Get Examples" button click
 */
function handleExamplesClick(event) {
  const button = event.currentTarget;
  const msgIndex = parseInt(button.dataset.msgIndex, 10);
  if (isNaN(msgIndex)) return;
  callAssistanceAPI("/api/get_examples", msgIndex, button);
}

// Attach event listeners to action buttons (refinement, examples)
document.querySelectorAll('.action-btn[data-action="refinement"]').forEach((btn) => {
  btn.addEventListener("click", handleRefinementClick);
});

document.querySelectorAll('.action-btn[data-action="examples"]').forEach((btn) => {
  btn.addEventListener("click", handleExamplesClick);
});

// ============== Multi-Select Key Points Extraction ==============

let selectionMode = false;
const selectedMessages = new Set();

const extractKeyPointsBtn = document.getElementById("extractKeyPointsBtn");
const selectionBar = document.getElementById("selectionBar");
const selectedCountEl = document.getElementById("selectedCount");
const confirmExtractBtn = document.getElementById("confirmExtractBtn");
const cancelSelectionBtn = document.getElementById("cancelSelectionBtn");
const keypointsModal = document.getElementById("keypointsModal");
const keypointsList = document.getElementById("keypointsList");
const closeKeypointsModal = document.getElementById("closeKeypointsModal");
const closeKeypointsBtn = document.getElementById("closeKeypointsBtn");
const copyKeypointsBtn = document.getElementById("copyKeypoints");

function updateSelectionUI() {
  selectedCountEl.textContent = selectedMessages.size;
  
  // Update checkbox states
  document.querySelectorAll(".expert-select-checkbox").forEach((cb) => {
    const idx = parseInt(cb.dataset.msgIndex, 10);
    cb.checked = selectedMessages.has(idx);
  });
  
  // Update card visual states
  document.querySelectorAll(".expert-row").forEach((row) => {
    const idx = parseInt(row.dataset.msgIndex, 10);
    row.classList.toggle("selected", selectedMessages.has(idx));
  });
}

function enterSelectionMode() {
  selectionMode = true;
  document.body.classList.add("selection-mode");
  selectionBar.classList.remove("hidden");
  extractKeyPointsBtn.classList.add("active");
  updateSelectionUI();
}

function exitSelectionMode() {
  selectionMode = false;
  selectedMessages.clear();
  document.body.classList.remove("selection-mode");
  selectionBar.classList.add("hidden");
  extractKeyPointsBtn.classList.remove("active");
  updateSelectionUI();
}

function toggleMessageSelection(index) {
  if (selectedMessages.has(index)) {
    selectedMessages.delete(index);
  } else {
    selectedMessages.add(index);
  }
  updateSelectionUI();
}

// Handle checkbox clicks
document.querySelectorAll(".expert-select-checkbox").forEach((cb) => {
  cb.addEventListener("change", (e) => {
    const idx = parseInt(e.target.dataset.msgIndex, 10);
    if (!isNaN(idx)) {
      if (e.target.checked) {
        selectedMessages.add(idx);
      } else {
        selectedMessages.delete(idx);
      }
      updateSelectionUI();
    }
  });
});

// Handle card clicks in selection mode
document.querySelectorAll(".expert-row").forEach((row) => {
  row.addEventListener("click", (e) => {
    if (!selectionMode) return;
    // Don't toggle if clicking on buttons or checkboxes
    if (e.target.closest("button") || e.target.closest("input") || e.target.closest(".action-btn")) return;
    
    const idx = parseInt(row.dataset.msgIndex, 10);
    if (!isNaN(idx)) {
      toggleMessageSelection(idx);
    }
  });
});

// Extract Key Points button in header
if (extractKeyPointsBtn) {
  extractKeyPointsBtn.addEventListener("click", () => {
    if (selectionMode) {
      exitSelectionMode();
    } else {
      enterSelectionMode();
    }
  });
}

// Cancel selection
if (cancelSelectionBtn) {
  cancelSelectionBtn.addEventListener("click", exitSelectionMode);
}

// Confirm extraction
if (confirmExtractBtn) {
  confirmExtractBtn.addEventListener("click", async () => {
    if (selectedMessages.size === 0) {
      alert("Please select at least one expert message.");
      return;
    }
    
    const indices = Array.from(selectedMessages);
    confirmExtractBtn.disabled = true;
    confirmExtractBtn.textContent = "Extracting...";
    
    try {
      const response = await fetch("/api/extract_keypoints", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ msg_indices: indices }),
      });
      
      const result = await response.json();
      
      if (result.success) {
        // Show modal with key points
        showKeypointsModal(result.data);
        exitSelectionMode();
      } else {
        alert(`Error: ${result.error || "Failed to extract key points"}`);
      }
    } catch (error) {
      alert(`Network error: ${error.message}`);
    } finally {
      confirmExtractBtn.disabled = false;
      confirmExtractBtn.textContent = "Extract Key Points";
    }
  });
}

function showKeypointsModal(keypoints) {
  if (!keypointsModal || !keypointsList) return;
  
  keypointsList.innerHTML = keypoints.map((kp) => `<li>${kp}</li>`).join("");
  keypointsModal.classList.remove("hidden");
}

function hideKeypointsModal() {
  if (keypointsModal) keypointsModal.classList.add("hidden");
}

if (closeKeypointsModal) closeKeypointsModal.addEventListener("click", hideKeypointsModal);
if (closeKeypointsBtn) closeKeypointsBtn.addEventListener("click", hideKeypointsModal);

// Copy key points to clipboard
if (copyKeypointsBtn) {
  copyKeypointsBtn.addEventListener("click", () => {
    const items = keypointsList.querySelectorAll("li");
    const text = Array.from(items).map((li, i) => `${i + 1}. ${li.textContent}`).join("\n");
    navigator.clipboard.writeText(text).then(() => {
      copyKeypointsBtn.textContent = "Copied!";
      setTimeout(() => {
        copyKeypointsBtn.textContent = "Copy to Clipboard";
      }, 2000);
    });
  });
}

// Close modal on overlay click
if (keypointsModal) {
  keypointsModal.querySelector(".modal-overlay")?.addEventListener("click", hideKeypointsModal);
}

// ============== Suggest Follow-ups (per expert message) ==============

function handleFollowupsClick(event) {
  event.preventDefault();
  event.stopPropagation();
  const button = event.currentTarget;
  const msgIndex = parseInt(button.dataset.msgIndex, 10);
  if (isNaN(msgIndex)) return;
  callAssistanceAPI("/api/suggest_followups", msgIndex, button);
}

document.querySelectorAll('.action-btn[data-action="followups"]').forEach((btn) => {
  btn.addEventListener("click", handleFollowupsClick);
});

// ============== HCI Mapping (map expert concepts to HCI domain) ==============

function handleHCIMappingClick(event) {
  event.preventDefault();
  event.stopPropagation();
  const button = event.currentTarget;
  const msgIndex = parseInt(button.dataset.msgIndex, 10);
  if (isNaN(msgIndex)) return;
  callAssistanceAPI("/api/hci_mapping", msgIndex, button);
}

document.querySelectorAll('.action-btn[data-action="hci-mapping"]').forEach((btn) => {
  btn.addEventListener("click", handleHCIMappingClick);
});

// ============== Flag as Mis-map ==============

function handleFlagMismapClick(event) {
  event.preventDefault();
  event.stopPropagation();
  const button = event.currentTarget;
  const msgIndex = button.dataset.msgIndex;
  
  // Create and submit a form programmatically
  const form = document.createElement("form");
  form.method = "POST";
  form.action = window.location.href;
  
  const actionInput = document.createElement("input");
  actionInput.type = "hidden";
  actionInput.name = "action";
  actionInput.value = "toggle_mismap";
  form.appendChild(actionInput);
  
  const indexInput = document.createElement("input");
  indexInput.type = "hidden";
  indexInput.name = "msg_index";
  indexInput.value = msgIndex;
  form.appendChild(indexInput);
  
  document.body.appendChild(form);
  form.submit();
}

document.querySelectorAll('.action-btn[data-action="flag-mismap"]').forEach((btn) => {
  btn.addEventListener("click", handleFlagMismapClick);
});

// ============== Collapsible Cards ==============

function handleCollapseToggle(event) {
  event.preventDefault();
  event.stopPropagation();
  const button = event.currentTarget;
  const card = button.closest(".collapsible");
  if (card) {
    card.classList.toggle("collapsed");
  }
}

document.querySelectorAll(".collapse-toggle").forEach((btn) => {
  btn.addEventListener("click", handleCollapseToggle);
});

// ============== Clickable Follow-up Questions ==============

function handleFollowupClick(event) {
  event.preventDefault();
  event.stopPropagation();
  
  const item = event.currentTarget;
  const question = item.dataset.question || item.textContent.replace(/^💭\s*/, "").trim();
  
  // Find the researcher input textarea
  const researcherInput = document.getElementById("researcherInput");
  if (researcherInput) {
    researcherInput.value = question;
    researcherInput.focus();
    
    // Visual feedback - show clicked state briefly
    item.classList.add("clicked");
    setTimeout(() => {
      item.classList.remove("clicked");
    }, 1500);
    
    // Scroll the input into view
    researcherInput.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

document.querySelectorAll(".clickable-followup").forEach((item) => {
  item.addEventListener("click", handleFollowupClick);
});
