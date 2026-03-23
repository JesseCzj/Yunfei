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

// ============== Cached Configurations ==============

const dsagCachedSection = document.getElementById("dsagCachedSection");
const dsagCachedSelect = document.getElementById("dsagCachedSelect");
const dsagLoadCachedBtn = document.getElementById("dsagLoadCachedBtn");
const dsagDividerText = document.getElementById("dsagDividerText");

async function loadCachedConfigs() {
  if (!dsagCachedSection) return;
  try {
    const resp = await fetch("/api/dsag/list_cached");
    const data = await resp.json();
    if (!data.success || !data.cached_configs || data.cached_configs.length === 0) {
      dsagCachedSection.style.display = "none";
      if (dsagDividerText) dsagDividerText.style.display = "none";
      return;
    }
    // Populate dropdown
    dsagCachedSelect.innerHTML = '<option value="">-- Select --</option>';
    data.cached_configs.forEach((cfg) => {
      const dateStr = cfg.saved_at ? cfg.saved_at.substring(0, 10) : "";
      const label = `${cfg.topic || "Untitled"} (${dateStr})`;
      const opt = document.createElement("option");
      opt.value = cfg.cache_key;
      opt.textContent = label;
      dsagCachedSelect.appendChild(opt);
    });
    dsagCachedSection.style.display = "block";
    if (dsagDividerText) dsagDividerText.style.display = "block";
  } catch (e) {
    console.error("Failed to load cached configs:", e);
  }
}
loadCachedConfigs();

if (dsagLoadCachedBtn) {
  dsagLoadCachedBtn.addEventListener("click", async () => {
    const cacheKey = dsagCachedSelect ? dsagCachedSelect.value : "";
    if (!cacheKey) {
      if (dsagInitStatus) dsagInitStatus.textContent = "Please select a configuration.";
      return;
    }
    dsagLoadCachedBtn.disabled = true;
    dsagLoadCachedBtn.textContent = "Loading...";
    if (dsagInitStatus) {
      dsagInitStatus.textContent = "Loading cached graph...";
      dsagInitStatus.className = "dsag-init-status dsag-status-building";
    }
    try {
      const resp = await fetch("/api/dsag/load_cached", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cache_key: cacheKey }),
      });
      const result = await resp.json();
      if (result.success) {
        if (dsagInitStatus) {
          dsagInitStatus.textContent = "Cached graph loaded. Ready!";
          dsagInitStatus.className = "dsag-init-status dsag-status-ready";
        }
        setTimeout(() => window.location.reload(), 800);
      } else {
        if (dsagInitStatus) {
          dsagInitStatus.textContent = `Error: ${result.error || "Failed to load"}`;
          dsagInitStatus.className = "dsag-init-status dsag-status-error";
        }
        dsagLoadCachedBtn.disabled = false;
        dsagLoadCachedBtn.textContent = "Load Selected";
      }
    } catch (err) {
      if (dsagInitStatus) {
        dsagInitStatus.textContent = `Network error: ${err.message}`;
        dsagInitStatus.className = "dsag-init-status dsag-status-error";
      }
      dsagLoadCachedBtn.disabled = false;
      dsagLoadCachedBtn.textContent = "Load Selected";
    }
  });
}

// ============== DSAG Init Form Submit ==============

if (dsagInitForm) {
  dsagInitForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const topic = dsagTopicInput ? dsagTopicInput.value.trim() : "";
    const researcherBg = dsagResearcherBgInput ? dsagResearcherBgInput.value.trim() : "";
    const expertBg = dsagExpertBgInput ? dsagExpertBgInput.value.trim() : "";
    const forceRebuild = document.getElementById("dsagForceRebuild")?.checked || false;

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
      dsagInitStatus.textContent = forceRebuild
        ? "Force rebuilding DSAG graph..."
        : "Generating DSAG graph...";
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
          force_rebuild: forceRebuild,
        }),
      });

      const result = await response.json();

      if (result.success) {
        sessionStorage.removeItem("dsagDraftTopic");
        sessionStorage.removeItem("dsagDraftResearcherBg");
        sessionStorage.removeItem("dsagDraftExpertBg");
        if (dsagInitStatus) {
          const sourceHint = result.cache_source
            ? ` (from ${result.cache_source})`
            : "";
          dsagInitStatus.textContent = result.cached
            ? `DSAG graph loaded from cache${sourceHint}. Ready!`
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
          dsagInitBtn.textContent = "Initialize";
        }
      }
    } catch (error) {
      if (dsagInitStatus) {
        dsagInitStatus.textContent = `Network error: ${error.message}`;
        dsagInitStatus.className = "dsag-init-status dsag-status-error";
      }
      if (dsagInitBtn) {
        dsagInitBtn.disabled = false;
        dsagInitBtn.textContent = "Initialize";
      }
    }
  });
}

// ============== Inline Expert Highlighting ==============
// Highlight jargon terms with blue tooltip chips and quoted DSAG evidence with yellow chips.

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function createPendingResearcherBubble(text) {
  if (!chatWindow || !text) return null;

  const row = document.createElement("div");
  row.className = "message-row researcher-row pending-message-row";
  row.innerHTML = `
    <div class="message-card researcher-card pending-message-card">
      <div class="message-header">
        <span class="message-role">Researcher</span>
        <span class="badge badge-pending">Sending...</span>
      </div>
      <div class="message-content">${escapeHtml(text)}</div>
    </div>
    <div class="message-avatar researcher-avatar">🧑‍💻</div>
  `;

  chatWindow.appendChild(row);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  sessionStorage.setItem("chatScrollPos", chatWindow.scrollTop.toString());
  return row;
}

function parseDatasetArray(rawValue) {
  if (!rawValue) return [];
  try {
    const parsed = JSON.parse(rawValue);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    return [];
  }
}

function collectQuoteRanges(text, quoteList) {
  const ranges = [];
  quoteList.forEach((item) => {
    const quote = String(item.text || "").trim();
    if (!quote) return;
    let startIndex = 0;
    while (startIndex < text.length) {
      const found = text.indexOf(quote, startIndex);
      if (found === -1) break;
      ranges.push({
        start: found,
        end: found + quote.length,
        type: "quote",
      });
      startIndex = found + quote.length;
    }
  });
  return ranges;
}

function collectJargonRanges(text, jargonList) {
  const ranges = [];
  const sorted = [...jargonList].sort(
    (a, b) => String(b.term || "").length - String(a.term || "").length,
  );
  sorted.forEach((item) => {
    const term = String(item.term || "").trim();
    const desc = String(item.desc || "");
    if (!term) return;
    const regex = new RegExp(`\\b${escapeRegExp(term)}\\b`, "gi");
    let match;
    while ((match = regex.exec(text)) !== null) {
      ranges.push({
        start: match.index,
        end: match.index + match[0].length,
        type: "jargon",
        desc,
      });
    }
  });
  return ranges;
}

function pickNonOverlappingRanges(ranges) {
  const sorted = [...ranges].sort((a, b) => {
    if (a.start !== b.start) return a.start - b.start;
    if (a.type !== b.type) return a.type === "quote" ? -1 : 1;
    return (b.end - b.start) - (a.end - a.start);
  });

  const picked = [];
  sorted.forEach((range) => {
    const overlaps = picked.some(
      (item) => !(range.end <= item.start || range.start >= item.end),
    );
    if (!overlaps) {
      picked.push(range);
    }
  });
  return picked.sort((a, b) => a.start - b.start);
}

function renderExpertHighlights(element) {
  const rawText = element.textContent || "";
  if (!rawText) return;

  const jargonList = parseDatasetArray(element.dataset.jargon);
  const quoteList = parseDatasetArray(element.dataset.quoteHighlights);
  const ranges = pickNonOverlappingRanges([
    ...collectQuoteRanges(rawText, quoteList),
    ...collectJargonRanges(rawText, jargonList),
  ]);

  if (ranges.length === 0) return;

  let html = "";
  let cursor = 0;
  ranges.forEach((range) => {
    if (cursor < range.start) {
      html += escapeHtml(rawText.slice(cursor, range.start));
    }

    const chunk = escapeHtml(rawText.slice(range.start, range.end));
    if (range.type === "quote") {
      html += `<span class="inline-quote-highlight">${chunk}</span>`;
    } else {
      html += `<span class="inline-jargon" data-tooltip="${escapeHtml(range.desc)}">${chunk}</span>`;
    }
    cursor = range.end;
  });

  if (cursor < rawText.length) {
    html += escapeHtml(rawText.slice(cursor));
  }
  
  element.innerHTML = html;
}

// Apply expert text highlighting to all expert messages
document.querySelectorAll(".expert-content").forEach(renderExpertHighlights);

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

// ============== DSAG Toggle Buttons (progressive disclosure) ==============

document.addEventListener("click", (event) => {
  const btn = event.target.closest(".dsag-toggle-btn");
  if (!btn) return;
  event.preventDefault();
  event.stopPropagation();

  const label = btn.textContent.trim();

  // For ConceptualGap: buttons are in a .dsag-toggle-row and content panels
  // are siblings of the row, matched by data-toggle-label attribute.
  const row = btn.closest(".dsag-toggle-row");
  if (row) {
    const parent = row.parentElement;
    const panel = parent.querySelector(
      `.dsag-toggle-content[data-toggle-label="${label}"]`
    );
    if (panel) {
      const isOpen = panel.style.display !== "none";
      panel.style.display = isOpen ? "none" : "block";
      btn.classList.toggle("active", !isOpen);
    }
    return;
  }

  // For TacitGap / ScopeGap: the content panel is the next sibling element.
  const content = btn.nextElementSibling;
  if (content && content.classList.contains("dsag-toggle-content")) {
    const isOpen = content.style.display !== "none";
    content.style.display = isOpen ? "none" : "block";
    btn.classList.toggle("active", !isOpen);
  }
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

// ============== Optimistic Researcher Send ==============

const optimisticResearcherForm = document.getElementById("researcherForm");
const optimisticResearcherInput = document.getElementById("researcherInput");

if (optimisticResearcherForm && optimisticResearcherInput) {
  optimisticResearcherForm.addEventListener("submit", () => {
    const text = optimisticResearcherInput.value.trim();
    if (!text) return;

    createPendingResearcherBubble(text);

    // Preserve the submitted text in a hidden field before clearing the
    // visible textarea, otherwise the browser may submit an empty value.
    let hiddenField = optimisticResearcherForm.querySelector(
      'input[data-optimistic-shadow="researcher"]'
    );
    if (!hiddenField) {
      hiddenField = document.createElement("input");
      hiddenField.type = "hidden";
      hiddenField.name = "researcher_input";
      hiddenField.setAttribute("data-optimistic-shadow", "researcher");
      optimisticResearcherForm.appendChild(hiddenField);
    }
    hiddenField.value = text;

    if (optimisticResearcherInput.name) {
      optimisticResearcherInput.dataset.originalName = optimisticResearcherInput.name;
      optimisticResearcherInput.name = "";
    }
    optimisticResearcherInput.value = "";

    const sendButton = optimisticResearcherForm.querySelector(".btn-send");
    if (sendButton) {
      sendButton.disabled = true;
      sendButton.textContent = "Sending...";
    }
  });
}

// ============== Transcript Summary Jump Links ==============

let transcriptJumpTimeout = null;

function clearTranscriptJumpHighlights() {
  document.querySelectorAll(".conversation-jump-highlight").forEach((el) => {
    el.classList.remove("conversation-jump-highlight");
  });
  document.querySelectorAll(".ts-sub-bullet-active").forEach((el) => {
    el.classList.remove("ts-sub-bullet-active");
  });
  if (transcriptJumpTimeout) {
    window.clearTimeout(transcriptJumpTimeout);
    transcriptJumpTimeout = null;
  }
}

function handleTranscriptSummaryJump(event) {
  event.preventDefault();
  event.stopPropagation();

  const item = event.currentTarget;
  const turn = String(item.dataset.targetTurn || "").trim();
  if (!turn) return;

  const targets = Array.from(
    document.querySelectorAll(`.message-row[data-turn-index="${turn}"]`)
  );
  if (targets.length === 0) return;

  clearTranscriptJumpHighlights();
  item.classList.add("ts-sub-bullet-active");
  targets.forEach((target) => target.classList.add("conversation-jump-highlight"));
  targets[0].scrollIntoView({ behavior: "smooth", block: "center" });

  transcriptJumpTimeout = window.setTimeout(() => {
    targets.forEach((target) => target.classList.remove("conversation-jump-highlight"));
    item.classList.remove("ts-sub-bullet-active");
    transcriptJumpTimeout = null;
  }, 2500);
}

document.querySelectorAll(".ts-sub-bullet-clickable").forEach((item) => {
  item.addEventListener("click", handleTranscriptSummaryJump);
});

// ============== Resizable Three-Panel Layout ==============

function initPanelResizers() {
  const container = document.querySelector(".main-panels");
  if (!container) return;

  const leftPanel = container.querySelector(".script-panel");
  const rightPanel = container.querySelector(".process-panel");
  const leftResizer = container.querySelector('.panel-resizer[data-resizer="left"]');
  const rightResizer = container.querySelector('.panel-resizer[data-resizer="right"]');
  if (!leftPanel || !rightPanel || !leftResizer || !rightResizer) return;

  const minLeft = 240;
  const minCenter = 420;
  const minRight = 240;
  const resizerWidth = 16; // two resizers, 8px each

  let dragSide = null;
  let startX = 0;
  let startLeftWidth = 0;
  let startRightWidth = 0;

  const stopDragging = () => {
    if (!dragSide) return;
    dragSide = null;
    leftResizer.classList.remove("is-dragging");
    rightResizer.classList.remove("is-dragging");
    document.body.classList.remove("is-resizing-panels");
    window.removeEventListener("mousemove", onDragMove);
    window.removeEventListener("mouseup", stopDragging);
  };

  const onDragMove = (event) => {
    if (!dragSide) return;
    const dx = event.clientX - startX;
    const totalWidth = container.clientWidth;

    if (dragSide === "left") {
      let nextLeft = startLeftWidth + dx;
      const maxLeft = totalWidth - startRightWidth - minCenter - resizerWidth;
      nextLeft = Math.max(minLeft, Math.min(maxLeft, nextLeft));
      container.style.setProperty("--left-panel-width", `${nextLeft}px`);
      return;
    }

    let nextRight = startRightWidth - dx;
    const leftCurrent = leftPanel.getBoundingClientRect().width;
    const maxRight = totalWidth - leftCurrent - minCenter - resizerWidth;
    nextRight = Math.max(minRight, Math.min(maxRight, nextRight));
    container.style.setProperty("--right-panel-width", `${nextRight}px`);
  };

  const startDragging = (side, event) => {
    if (window.innerWidth <= 1024) return;
    dragSide = side;
    startX = event.clientX;
    startLeftWidth = leftPanel.getBoundingClientRect().width;
    startRightWidth = rightPanel.getBoundingClientRect().width;

    document.body.classList.add("is-resizing-panels");
    (side === "left" ? leftResizer : rightResizer).classList.add("is-dragging");
    window.addEventListener("mousemove", onDragMove);
    window.addEventListener("mouseup", stopDragging);
    event.preventDefault();
  };

  leftResizer.addEventListener("mousedown", (event) => startDragging("left", event));
  rightResizer.addEventListener("mousedown", (event) => startDragging("right", event));

  window.addEventListener("blur", stopDragging);
}

initPanelResizers();
