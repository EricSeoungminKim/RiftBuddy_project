const BACKEND_WS_URL = "ws://localhost:8765";

let currentState = normalizeState({});
let wsClient = null;

const connectionStatus = document.querySelector("#connectionStatus");
const clickModeButton = document.querySelector("#clickModeButton");
const miaList = document.querySelector("#miaList");
const spellGrid = document.querySelector("#spellGrid");
const minimapDebugPanel = document.querySelector("#minimapDebugPanel");
const debugRegion = document.querySelector("#debugRegion");
const debugFrame = document.querySelector("#debugFrame");
const debugPreview = document.querySelector("#debugPreview");
const debugScores = document.querySelector("#debugScores");
const calibrationBox = document.querySelector("#calibrationBox");

function renderConnectionStatus(status) {
  connectionStatus.textContent = status;
  connectionStatus.dataset.status = status;
}

function renderMiaList() {
  miaList.innerHTML = "";

  if (currentState.gameStatus.phase === "waiting_for_active_game") {
    miaList.append(emptyState("waiting"));
    return;
  }

  if (currentState.mia.length === 0) {
    miaList.append(emptyState("clear"));
    return;
  }

  currentState.mia.slice(0, 5).forEach((item) => {
    const loadout = findLoadout(item.champion);
    const row = document.createElement("div");
    row.className = "mia-row";
    row.innerHTML = `
      <span class="champion-token">
        ${roleBadge(loadout?.role)}${item.champion}
      </span>
      <strong>${formatSeconds(item.elapsed_sec)}</strong>
    `;
    miaList.append(row);
  });
}

function renderSpellGrid() {
  spellGrid.innerHTML = "";

  if (currentState.enemyLoadout.length === 0) {
    spellGrid.append(emptyState(currentState.gameStatus.message || "waiting"));
    return;
  }

  currentState.enemyLoadout.forEach((loadout) => {
    const row = document.createElement("div");
    row.className = "spell-row";

    const label = document.createElement("div");
    label.className = "champion-label";
    label.innerHTML = `${roleBadge(loadout.role)}${loadout.champion}`;
    row.append(label);

    loadout.spells.forEach((spell) => {
      const state = findSpellState(currentState, loadout.champion, spell);
      const button = document.createElement("button");
      button.className = "spell-button";
      button.type = "button";
      button.dataset.champion = loadout.champion;
      button.dataset.spell = spell;
      button.title = "Left click: start timer. Right click: mark ready.";
      button.innerHTML = `
        <span class="spell-mark">${spell.slice(0, 1)}</span>
        <span class="spell-name">${spell}</span>
        <strong>${state ? formatSeconds(state.remaining_sec) : "ready"}</strong>
      `;
      button.addEventListener("click", () => handleSpellClick(loadout.champion, spell));
      button.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        handleSpellClear(loadout.champion, spell);
      });
      row.append(button);
    });

    spellGrid.append(row);
  });
}

function emptyState(text) {
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.textContent = text;
  return empty;
}

function findLoadout(champion) {
  return currentState.enemyLoadout.find((item) => item.champion === champion);
}

function roleBadge(role) {
  const label = roleLabel(role);
  return label ? `<span class="role-badge">${label}</span>` : "";
}

function roleLabel(role) {
  switch (role) {
    case "TOP":
      return "TOP";
    case "JUNGLE":
      return "JG";
    case "MIDDLE":
      return "MID";
    case "BOTTOM":
      return "ADC";
    case "UTILITY":
      return "SUP";
    default:
      return "";
  }
}

function renderState(payload) {
  currentState = normalizeState(payload);
  renderMiaList();
  renderSpellGrid();
  renderMinimapDebug();
}

function handleSpellClick(champion, spell) {
  const sent = wsClient.sendSpellClicked(champion, spell);
  if (!sent) {
    renderConnectionStatus("offline");
  }
}

function handleSpellClear(champion, spell) {
  const sent = wsClient.sendSpellCleared(champion, spell);
  if (!sent) {
    renderConnectionStatus("offline");
  }
}

function renderMinimapDebug() {
  const debug = currentState.minimapDebug;
  const visible = debug.enabled || debug.calibrationMode;
  minimapDebugPanel.hidden = !visible;
  calibrationBox.hidden = !(visible && debug.calibrationMode && debug.region);

  if (!visible) {
    return;
  }

  renderCalibrationBox(debug);
  renderDebugPanel(debug);
}

function renderCalibrationBox(debug) {
  if (!debug.region) {
    return;
  }

  const region = captureRegionToCssRegion(
    debug.region,
    debug.captureDisplay,
    { width: window.innerWidth, height: window.innerHeight },
  );

  calibrationBox.style.left = `${region.left}px`;
  calibrationBox.style.top = `${region.top}px`;
  calibrationBox.style.width = `${region.width}px`;
  calibrationBox.style.height = `${region.height}px`;
}

function renderDebugPanel(debug) {
  debugRegion.textContent = debug.region
    ? `region ${debug.region.left},${debug.region.top} ${debug.region.width}x${debug.region.height}`
    : "region unavailable";
  debugFrame.textContent = debug.frameSize
    ? `frame ${debug.frameSize.width}x${debug.frameSize.height}`
    : "frame waiting";
  debugPreview.hidden = !debug.capturePreview;
  if (debug.capturePreview) {
    debugPreview.src = debug.capturePreview;
  }

  debugScores.innerHTML = "";
  debug.champions.forEach((item) => {
    const row = document.createElement("div");
    row.className = "debug-score-row";
    row.dataset.detected = String(item.detected === true);
    row.innerHTML = `
      <span>${item.champion}</span>
      <span class="debug-score-state">${item.detected === true ? "seen" : "search"}</span>
      <strong>${formatConfidence(item.confidence)}</strong>
    `;
    debugScores.append(row);
  });
}

function formatConfidence(value) {
  const confidence = Number(value);
  if (!Number.isFinite(confidence)) {
    return "0.000";
  }
  return confidence.toFixed(3);
}

function initializeCalibrationControls() {
  minimapDebugPanel.addEventListener("click", (event) => {
    const button = event.target.closest("[data-region-key]");
    if (!button) {
      return;
    }

    const key = button.dataset.regionKey;
    const value = Number(button.dataset.regionValue);
    if (!key || !Number.isFinite(value)) {
      return;
    }

    const sent = wsClient.sendMinimapRegionAdjusted({ [key]: value });
    if (!sent) {
      renderConnectionStatus("offline");
    }
  });
}

async function initializeClickMode() {
  if (!window.riftBuddyOverlay) {
    return;
  }

  const isClickThrough = await window.riftBuddyOverlay.getClickThrough();
  renderClickMode(isClickThrough);
  window.riftBuddyOverlay.onClickThroughChanged(renderClickMode);

  clickModeButton.addEventListener("click", async () => {
    const nextMode = clickModeButton.dataset.clickThrough !== "true";
    const enabled = await window.riftBuddyOverlay.setClickThrough(nextMode);
    renderClickMode(enabled);
  });
}

function initializeSpellHotkeys() {
  if (!window.riftBuddyOverlay) {
    return;
  }

  window.riftBuddyOverlay.onSpellHotkey((binding) => {
    const action = resolveSpellHotkeyAction(currentState, binding);
    if (!action) {
      return;
    }

    const sent =
      action.type === "spell_cleared"
        ? wsClient.sendSpellCleared(action.champion, action.spell)
        : wsClient.sendSpellClicked(action.champion, action.spell);
    if (!sent) {
      renderConnectionStatus("offline");
    }
  });
}

function renderClickMode(isClickThrough) {
  clickModeButton.dataset.clickThrough = String(isClickThrough);
  clickModeButton.textContent = isClickThrough ? "pass-through" : "click mode";
}

function initialize() {
  renderState({ mia: [], spells: [] });
  initializeClickMode();
  initializeSpellHotkeys();
  initializeCalibrationControls();

  wsClient = new window.RiftBuddyWsClient({
    url: BACKEND_WS_URL,
    onState: renderState,
    onStatus: renderConnectionStatus
  });
  wsClient.connect();
}

initialize();
