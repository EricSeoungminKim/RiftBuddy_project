const {
  app,
  BrowserWindow,
  globalShortcut,
  ipcMain,
  screen,
} = require("electron");
const path = require("path");
const { buildOverlayWindowOptions } = require("./window-options");

const TOGGLE_CLICK_THROUGH_SHORTCUT = "CommandOrControl+Shift+X";

let overlayWindow = null;
let isClickThrough = false;

function createOverlayWindow() {
  const display = screen.getPrimaryDisplay();
  const { x, y, width, height } = display.bounds;

  overlayWindow = new BrowserWindow(
    buildOverlayWindowOptions({
      bounds: { x, y, width, height },
      preloadPath: path.join(__dirname, "preload.js"),
    }),
  );

  pinOverlayWindow();
  overlayWindow.loadFile(path.join(__dirname, "overlay", "index.html"));

  overlayWindow.on("closed", () => {
    overlayWindow = null;
  });
}

function setClickThrough(enabled) {
  isClickThrough = enabled;
  if (!overlayWindow) {
    return;
  }

  overlayWindow.setIgnoreMouseEvents(enabled, { forward: true });
  overlayWindow.webContents.send("click-through-changed", enabled);
}

function toggleClickThrough() {
  setClickThrough(!isClickThrough);
}

function pinOverlayWindow() {
  if (!overlayWindow) {
    return;
  }

  overlayWindow.setAlwaysOnTop(true, "screen-saver", 1);
  overlayWindow.setFocusable(false);
  overlayWindow.setVisibleOnAllWorkspaces(true, {
    visibleOnFullScreen: true,
    skipTransformProcessType: true,
  });
  overlayWindow.moveTop();
}

function getDisplayMetrics() {
  const display = screen.getPrimaryDisplay();
  return {
    bounds: display.bounds,
    workArea: display.workArea,
    scaleFactor: display.scaleFactor,
    windowBounds: overlayWindow ? overlayWindow.getBounds() : null,
  };
}

app.whenReady().then(() => {
  createOverlayWindow();
  globalShortcut.register(TOGGLE_CLICK_THROUGH_SHORTCUT, toggleClickThrough);

  ipcMain.handle("overlay:get-click-through", () => isClickThrough);
  ipcMain.handle("overlay:set-click-through", (_event, enabled) => {
    setClickThrough(Boolean(enabled));
    return isClickThrough;
  });
  ipcMain.handle("overlay:get-display-metrics", getDisplayMetrics);

  screen.on("display-metrics-changed", () => {
    if (!overlayWindow) {
      return;
    }
    overlayWindow.webContents.send("display-metrics-changed", getDisplayMetrics());
    pinOverlayWindow();
  });

  setInterval(pinOverlayWindow, 1000);
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("will-quit", () => {
  globalShortcut.unregisterAll();
});
