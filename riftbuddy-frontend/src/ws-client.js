class RiftBuddyWsClient {
  constructor({ url, onState, onStatus }) {
    this.url = url;
    this.onState = onState;
    this.onStatus = onStatus;
    this.socket = null;
    this.reconnectTimer = null;
  }

  connect() {
    this._setStatus("connecting");
    this.socket = new WebSocket(this.url);

    this.socket.addEventListener("open", () => {
      this._setStatus("connected");
    });

    this.socket.addEventListener("message", (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "state_update") {
        this.onState(payload);
      }
    });

    this.socket.addEventListener("close", () => {
      this._setStatus("disconnected");
      this._scheduleReconnect();
    });

    this.socket.addEventListener("error", () => {
      this._setStatus("error");
      this.socket.close();
    });
  }

  sendSpellClicked(champion, spell) {
    return this._sendSpellMessage("spell_clicked", champion, spell);
  }

  sendSpellCleared(champion, spell) {
    return this._sendSpellMessage("spell_cleared", champion, spell);
  }

  sendMinimapRegionAdjusted(delta) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return false;
    }

    this.socket.send(JSON.stringify({ type: "minimap_region_adjusted", ...delta }));
    return true;
  }

  _sendSpellMessage(type, champion, spell) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return false;
    }

    this.socket.send(JSON.stringify({ type, champion, spell }));
    return true;
  }

  _scheduleReconnect() {
    if (this.reconnectTimer) {
      return;
    }

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 1000);
  }

  _setStatus(status) {
    this.onStatus(status);
  }
}

window.RiftBuddyWsClient = RiftBuddyWsClient;
