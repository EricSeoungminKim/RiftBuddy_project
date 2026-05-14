const test = require("node:test");
const assert = require("node:assert/strict");

const {
  DEFAULT_ENEMY_LOADOUT,
  captureRegionToCssRegion,
  findSpellState,
  formatSeconds,
  normalizeState,
} = require("./state");

test("normalizeState returns safe arrays", () => {
  assert.deepEqual(normalizeState({}), {
    enemyLoadout: DEFAULT_ENEMY_LOADOUT,
    gameStatus: { phase: "test_mode", message: "" },
    mia: [],
    minimapDebug: {},
    spells: [],
  });
});

test("normalizeState uses backend enemy loadout when present", () => {
  assert.deepEqual(
    normalizeState({
      enemy_loadout: [
        { champion: "Milio", role: "UTILITY", spells: ["Flash", "Heal"] },
        { champion: "Garen", role: "TOP", spells: ["Flash", "Teleport"] },
      ],
    }).enemyLoadout,
    [
      { champion: "Garen", role: "TOP", spells: ["Flash", "Teleport"] },
      { champion: "Milio", role: "UTILITY", spells: ["Flash", "Heal"] },
    ],
  );
});

test("normalizeState hides default loadout while waiting for active game", () => {
  assert.deepEqual(
    normalizeState({
      game_status: {
        phase: "waiting_for_active_game",
        message: "Waiting for active League game",
      },
      enemy_loadout: [],
    }),
    {
      enemyLoadout: [],
      gameStatus: {
        phase: "waiting_for_active_game",
        message: "Waiting for active League game",
      },
      mia: [],
      minimapDebug: {},
      spells: [],
    },
  );
});

test("normalizeState keeps minimap debug data", () => {
  assert.deepEqual(
    normalizeState({
      minimap_debug: {
        enabled: true,
        calibration_mode: true,
        auto_region: true,
        region: { top: 790, left: 1630, width: 290, height: 290 },
        capture_display: { top: 0, left: 0, width: 4096, height: 2304 },
        capture_preview: "data:image/jpeg;base64,abc",
        champions: [{ champion: "Garen", confidence: 0.42, detected: false }],
      },
    }).minimapDebug,
    {
      enabled: true,
      calibrationMode: true,
      autoRegion: true,
      region: { top: 790, left: 1630, width: 290, height: 290 },
      captureDisplay: { top: 0, left: 0, width: 4096, height: 2304 },
      frameSize: null,
      capturePath: null,
      capturePreview: "data:image/jpeg;base64,abc",
      threshold: null,
      updatedAt: null,
      champions: [{ champion: "Garen", confidence: 0.42, detected: false }],
    },
  );
});

test("captureRegionToCssRegion maps capture pixels into overlay pixels", () => {
  assert.deepEqual(
    captureRegionToCssRegion(
      { top: 790, left: 1630, width: 290, height: 290 },
      { top: 0, left: 0, width: 4096, height: 2304 },
      { width: 2048, height: 1152 },
    ),
    { top: 395, left: 815, width: 145, height: 145 },
  );
});

test("findSpellState returns matching champion spell timer", () => {
  const state = normalizeState({
    spells: [{ champion: "Jinx", spell: "Flash", remaining_sec: 244 }],
  });

  assert.deepEqual(findSpellState(state, "Jinx", "Flash"), {
    champion: "Jinx",
    spell: "Flash",
    remaining_sec: 244,
  });
});

test("formatSeconds renders short and minute timers", () => {
  assert.equal(formatSeconds(12.2), "13s");
  assert.equal(formatSeconds(125), "2:05");
});
