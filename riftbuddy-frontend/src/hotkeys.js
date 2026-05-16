const SPELL_HOTKEY_BINDINGS = Array.from({ length: 5 }, (_, index) => {
  const laneNumber = index + 1;
  return [
    {
      accelerator: `CommandOrControl+Alt+${laneNumber}`,
      loadoutIndex: index,
      spellIndex: 0,
    },
    {
      accelerator: `CommandOrControl+Alt+Shift+${laneNumber}`,
      loadoutIndex: index,
      spellIndex: 1,
    },
  ];
}).flat();

function spellHotkeyEventName() {
  return "spell-hotkey";
}

function resolveSpellHotkeyAction(state, hotkey) {
  const loadout = state.enemyLoadout[hotkey.loadoutIndex];
  if (!loadout) {
    return null;
  }

  const spell = loadout.spells[hotkey.spellIndex];
  if (!spell) {
    return null;
  }

  const activeTimer = state.spells.find(
    (item) => item.champion === loadout.champion && item.spell === spell,
  );

  return {
    type: activeTimer ? "spell_cleared" : "spell_clicked",
    champion: loadout.champion,
    spell,
  };
}

if (typeof window !== "undefined") {
  window.SPELL_HOTKEY_BINDINGS = SPELL_HOTKEY_BINDINGS;
  window.resolveSpellHotkeyAction = resolveSpellHotkeyAction;
  window.spellHotkeyEventName = spellHotkeyEventName;
}

if (typeof module !== "undefined") {
  module.exports = {
    SPELL_HOTKEY_BINDINGS,
    resolveSpellHotkeyAction,
    spellHotkeyEventName,
  };
}
