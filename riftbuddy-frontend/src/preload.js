const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("riftBuddyOverlay", {
  getClickThrough: () => ipcRenderer.invoke("overlay:get-click-through"),
  setClickThrough: (enabled) =>
    ipcRenderer.invoke("overlay:set-click-through", enabled),
  getDisplayMetrics: () => ipcRenderer.invoke("overlay:get-display-metrics"),
  onClickThroughChanged: (callback) => {
    ipcRenderer.on("click-through-changed", (_event, enabled) =>
      callback(enabled),
    );
  },
  onDisplayMetricsChanged: (callback) => {
    ipcRenderer.on("display-metrics-changed", (_event, metrics) =>
      callback(metrics),
    );
  },
});
