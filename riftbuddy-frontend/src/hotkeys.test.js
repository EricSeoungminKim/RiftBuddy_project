const test = require("node:test");
const assert = require("node:assert/strict");

const {
  SPELL_HOTKEY_BINDINGS,
  resolveSpellHotkeyAction,
  spellHotkeyEventName,
} = require("./hotkeys");

test("spell hotkey bindings map lanes to spell slots", () => {
  assert.deepEqual(SPELL_HOTKEY_BINDINGS.slice(0, 4), [
    {
      accelerator: "CommandOrControl+Alt+1",
      loadoutIndex: 0,
      spellIndex: 0,
    },
    {
      accelerator: "CommandOrControl+Alt+Shift+1",
      loadoutIndex: 0,
      spellIndex: 1,
    },
    {
      accelerator: "CommandOrControl+Alt+2",
      loadoutIndex: 1,
      spellIndex: 0,
    },
    {
      accelerator: "CommandOrControl+Alt+Shift+2",
      loadoutIndex: 1,
      spellIndex: 1,
    },
  ]);
  assert.equal(SPELL_HOTKEY_BINDINGS.length, 10);
  assert.equal(spellHotkeyEventName(), "spell-hotkey");
});

test("spell hotkey starts cooldown when timer is not active", () => {
  const action = resolveSpellHotkeyAction(
    {
      enemyLoadout: [
        { champion: "Garen", spells: ["Flash", "Teleport"] },
      ],
      spells: [],
    },
    { loadoutIndex: 0, spellIndex: 0 },
  );

  assert.deepEqual(action, {
    type: "spell_clicked",
    champion: "Garen",
    spell: "Flash",
  });
});

test("same spell hotkey clears cooldown when timer is already active", () => {
  const action = resolveSpellHotkeyAction(
    {
      enemyLoadout: [
        { champion: "Garen", spells: ["Flash", "Teleport"] },
      ],
      spells: [{ champion: "Garen", spell: "Flash", remaining_sec: 230 }],
    },
    { loadoutIndex: 0, spellIndex: 0 },
  );

  assert.deepEqual(action, {
    type: "spell_cleared",
    champion: "Garen",
    spell: "Flash",
  });
});

test("spell hotkey ignores missing loadout or spell slot", () => {
  assert.equal(
    resolveSpellHotkeyAction(
      { enemyLoadout: [], spells: [] },
      { loadoutIndex: 4, spellIndex: 0 },
    ),
    null,
  );
  assert.equal(
    resolveSpellHotkeyAction(
      { enemyLoadout: [{ champion: "Garen", spells: ["Flash"] }], spells: [] },
      { loadoutIndex: 0, spellIndex: 1 },
    ),
    null,
  );
});
