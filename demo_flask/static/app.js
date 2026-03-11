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

// ============== DSAG Initialization ==============

const dsagInitForm = document.getElementById("dsagInitForm");
const dsagInitBtn = document.getElementById("dsagInitBtn");
const dsagInitStatus = document.getElementById("dsagInitStatus");
const dsagTopicInput = document.getElementById("dsagTopic");
const dsagResearcherBgInput = document.getElementById("dsagResearcherBg");
const dsagExpertBgInput = document.getElementById("dsagExpertBg");

function restoreDsagDraft() {
  if (dsagTopicInput) dsagTopicInput.value = sessionStorage.getItem("dsagDraftTopic") || "";
  if (dsagResearcherBgInput) dsagResearcherBgInput.value = sessionStorage.getItem("dsagDraftResearcherBg") || "";
  if (dsagExpertBgInput) dsagExpertBgInput.value = sessionStorage.getItem("dsagDraftExpertBg") || "";
}

function persistDsagDraft() {
  if (dsagTopicInput) sessionStorage.setItem("dsagDraftTopic", dsagTopicInput.value || "");
  if (dsagResearcherBgInput) sessionStorage.setItem("dsagDraftResearcherBg", dsagResearcherBgInput.value || "");
  if (dsagExpertBgInput) sessionStorage.setItem("dsagDraftExpertBg", dsagExpertBgInput.value || "");
}

restoreDsagDraft();
if (dsagTopicInput) dsagTopicInput.addEventListener("input", persistDsagDraft);
if (dsagResearcherBgInput) dsagResearcherBgInput.addEventListener("input", persistDsagDraft);
if (dsagExpertBgInput) dsagExpertBgInput.addEventListener("input", persistDsagDraft);

if (dsagInitForm) {
  dsagInitForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const topic = dsagTopicInput ? dsagTopicInput.value.trim() : "";
    const researcherBg = dsagResearcherBgInput ? dsagResearcherBgInput.value.trim() : "";
    const expertBg = dsagExpertBgInput ? dsagExpertBgInput.value.trim() : "";

    if (!topic || !researcherBg || !expertBg) {
      if (dsagInitStatus) dsagInitStatus.textContent = "All fields are required.";
      return;
    }

    // Disable button and show loading
    if (dsagInitBtn) {
      dsagInitBtn.disabled = true;
      dsagInitBtn.textContent = "Building graph...";
    }
    if (dsagInitStatus) {
      dsagInitStatus.textContent = "Generating DSAG graph...";
      dsagInitStatus.className = "dsag-init-status dsag-status-building";
    }

    try {
      const response = await fetch("/api/dsag/init", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: topic,
          researcher_bg: researcherBg,
          expert_bg: expertBg,
        }),
      });

      const result = await response.json();

      if (result.success) {
        sessionStorage.removeItem("dsagDraftTopic");
        sessionStorage.removeItem("dsagDraftResearcherBg");
        sessionStorage.removeItem("dsagDraftExpertBg");
        if (dsagInitStatus) {
          dsagInitStatus.textContent = result.cached
            ? "DSAG graph loaded from cache. Ready!"
            : "DSAG graph built successfully. Ready!";
          dsagInitStatus.className = "dsag-init-status dsag-status-ready";
        }
        // Reload page to update the status badge and enable auto-analysis
        setTimeout(() => window.location.reload(), 1000);
      } else {
        if (dsagInitStatus) {
          dsagInitStatus.textContent = `Error: ${result.error || "Failed to build graph"}`;
          dsagInitStatus.className = "dsag-init-status dsag-status-error";
        }
        if (dsagInitBtn) {
          dsagInitBtn.disabled = false;
          dsagInitBtn.textContent = "Initialize DSAG";
        }
      }
    } catch (error) {
      if (dsagInitStatus) {
        dsagInitStatus.textContent = `Network error: ${error.message}`;
        dsagInitStatus.className = "dsag-init-status dsag-status-error";
      }
      if (dsagInitBtn) {
        dsagInitBtn.disabled = false;
        dsagInitBtn.textContent = "Initialize DSAG";
      }
    }
  });
}

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
  const uploadForm = fileInput.closest("form");
  fileInput.addEventListener("change", () => {
    const selectedFile = fileInput.files[0];
    fileLabel.textContent = fileInput.files[0]?.name || "Upload interview script (txt, docx, pdf)";
    if (selectedFile && uploadForm) {
      persistDsagDraft();
      saveScrollPosition();
      uploadForm.submit();
    }
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

// Legacy llm_backend on-demand actions removed; DSAG-only UI now.

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
