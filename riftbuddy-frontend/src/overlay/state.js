const DEFAULT_ENEMY_LOADOUT = [
  { champion: "Jayce", role: "TOP", spells: ["Flash", "Teleport"] },
  { champion: "LeeSin", role: "JUNGLE", spells: ["Flash", "Smite"] },
  { champion: "Zed", role: "MIDDLE", spells: ["Flash", "Ignite"] },
  { champion: "Jinx", role: "BOTTOM", spells: ["Flash", "Heal"] },
  { champion: "Thresh", role: "UTILITY", spells: ["Flash", "Ignite"] },
];

const ROLE_ORDER = {
  TOP: 0,
  JUNGLE: 1,
  MIDDLE: 2,
  BOTTOM: 3,
  UTILITY: 4,
};

function normalizeState(payload) {
  const gameStatus = normalizeGameStatus(payload.game_status);
  return {
    enemyLoadout: normalizeEnemyLoadout(payload.enemy_loadout, gameStatus),
    gameStatus,
    mia: Array.isArray(payload.mia) ? payload.mia : [],
    minimapDebug: normalizeMinimapDebug(payload.minimap_debug),
    spells: normalizeSpells(payload.spells),
  };
}

function normalizeGameStatus(gameStatus) {
  if (!gameStatus || typeof gameStatus !== "object" || Array.isArray(gameStatus)) {
    return { phase: "test_mode", message: "" };
  }

  return {
    phase:
      typeof gameStatus.phase === "string" && gameStatus.phase.length > 0
        ? gameStatus.phase
        : "test_mode",
    message:
      typeof gameStatus.message === "string" ? gameStatus.message : "",
  };
}

function normalizeEnemyLoadout(enemyLoadout, gameStatus = normalizeGameStatus(null)) {
  if (gameStatus.phase === "waiting_for_active_game") {
    return [];
  }

  if (!Array.isArray(enemyLoadout) || enemyLoadout.length === 0) {
    return DEFAULT_ENEMY_LOADOUT;
  }

  return enemyLoadout
    .filter((item) => item && typeof item.champion === "string")
    .map((item) => ({
      champion: item.champion,
      role: normalizeRole(item.role),
      spells: Array.isArray(item.spells)
        ? item.spells.filter((spell) => typeof spell === "string")
        : [],
    }))
    .filter((item) => item.spells.length > 0)
    .sort(compareLoadoutRole);
}

function normalizeRole(role) {
  if (typeof role !== "string" || role.length === 0) {
    return null;
  }
  return role.toUpperCase();
}

function compareLoadoutRole(left, right) {
  const leftOrder = ROLE_ORDER[left.role] ?? Number.MAX_SAFE_INTEGER;
  const rightOrder = ROLE_ORDER[right.role] ?? Number.MAX_SAFE_INTEGER;
  return leftOrder - rightOrder;
}

function normalizeMinimapDebug(debug) {
  if (!debug || typeof debug !== "object" || Array.isArray(debug)) {
    return {};
  }

  return {
    enabled: debug.enabled === true,
    calibrationMode: debug.calibration_mode === true,
    autoRegion: debug.auto_region === true,
    region: normalizeRegion(debug.region),
    captureDisplay: normalizeRegion(debug.capture_display),
    frameSize: normalizeFrameSize(debug.frame_size),
    capturePath:
      typeof debug.capture_path === "string" ? debug.capture_path : null,
    capturePreview:
      typeof debug.capture_preview === "string" ? debug.capture_preview : null,
    threshold: typeof debug.threshold === "number" ? debug.threshold : null,
    updatedAt: typeof debug.updated_at === "number" ? debug.updated_at : null,
    champions: Array.isArray(debug.champions)
      ? debug.champions.filter((item) => item && typeof item.champion === "string")
      : [],
  };
}

function normalizeRegion(region) {
  if (!region || typeof region !== "object" || Array.isArray(region)) {
    return null;
  }

  const { top, left, width, height } = region;
  if ([top, left, width, height].some((value) => typeof value !== "number")) {
    return null;
  }

  return { top, left, width, height };
}

function normalizeFrameSize(frameSize) {
  if (!frameSize || typeof frameSize !== "object" || Array.isArray(frameSize)) {
    return null;
  }

  const { width, height } = frameSize;
  if (typeof width !== "number" || typeof height !== "number") {
    return null;
  }

  return { width, height };
}

function normalizeSpells(spells) {
  if (!Array.isArray(spells)) {
    return [];
  }

  return spells.filter((item) => {
    if (!item || typeof item !== "object") {
      return false;
    }
    return Number(item.remaining_sec) > 0;
  });
}

function captureRegionToCssRegion(region, captureDisplay, viewport) {
  if (!region || !captureDisplay || !viewport) {
    return region;
  }

  const scaleX = viewport.width / captureDisplay.width;
  const scaleY = viewport.height / captureDisplay.height;
  return {
    top: Math.round((region.top - captureDisplay.top) * scaleY),
    left: Math.round((region.left - captureDisplay.left) * scaleX),
    width: Math.round(region.width * scaleX),
    height: Math.round(region.height * scaleY),
  };
}

function findSpellState(state, champion, spell) {
  return state.spells.find(
    (item) => item.champion === champion && item.spell === spell,
  );
}

function formatSeconds(value) {
  const seconds = Math.max(0, Math.ceil(Number(value) || 0));
  if (seconds >= 60) {
    const minutes = Math.floor(seconds / 60);
    const remainder = seconds % 60;
    return `${minutes}:${String(remainder).padStart(2, "0")}`;
  }
  return `${seconds}s`;
}

if (typeof module !== "undefined") {
  module.exports = {
    DEFAULT_ENEMY_LOADOUT,
    captureRegionToCssRegion,
    findSpellState,
    formatSeconds,
    normalizeMinimapDebug,
    normalizeEnemyLoadout,
    normalizeGameStatus,
    normalizeState,
    normalizeSpells,
  };
}
