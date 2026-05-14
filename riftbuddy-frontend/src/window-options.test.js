const test = require("node:test");
const assert = require("node:assert/strict");

const { buildOverlayWindowOptions } = require("./window-options");

test("overlay window is non-focusable so clicks do not steal game focus", () => {
  const options = buildOverlayWindowOptions({
    bounds: { x: 10, y: 20, width: 1280, height: 720 },
    preloadPath: "/tmp/preload.js",
  });

  assert.equal(options.focusable, false);
  assert.equal(options.width, 1280);
  assert.equal(options.height, 720);
  assert.equal(options.x, 10);
  assert.equal(options.y, 20);
  assert.equal(options.webPreferences.preload, "/tmp/preload.js");
});
