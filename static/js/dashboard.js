/* dashboard.js — preview stream, PTZ/zoom controls, pairing, face capture */

const $ = (id) => document.getElementById(id);

async function post(url, body, retries = 0) {
  for (;;) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const err = new Error(data.error || `Request failed (${res.status})`);
        err.status = res.status;
        throw err;
      }
      return data;
    } catch (e) {
      /* A server response (has .status) is final. A network-level failure
         ("Failed to fetch", no .status) may be the Pi's shared Wi-Fi/BT
         radio stalling Wi-Fi during Bluetooth pairing — retry those. */
      if (e.status !== undefined || retries-- <= 0) throw e;
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
}

/* ── Live preview ────────────────────────────────────────────────────── */
const socket = io();
const img = $("preview");
const placeholder = $("preview-placeholder");

/* Closing the tab must register as a sign-out promptly: an abandoned
   connection can take the whole Socket.IO ping timeout to die, and the
   camera keeps tracking that entire time. pagehide also fires on
   refresh/navigation, but the server's grace period absorbs those.
   pageshow restores the socket when the page comes back from bfcache
   (a manual disconnect() disables auto-reconnect). */
addEventListener("pagehide", () => socket.disconnect());
addEventListener("pageshow", (e) => { if (e.persisted) socket.connect(); });

socket.on("preview_frame", (msg) => {
  img.src = "data:image/jpeg;base64," + msg.image;
  if (img.hidden) {
    img.hidden = false;
    placeholder.hidden = true;
  }
  paintTelemetry(msg.telemetry);
});

socket.on("session_error", (msg) => {
  placeholder.textContent = "Camera unavailable — " + msg.error;
});

function paintTelemetry(t) {
  if (!t) return;
  $("t-fps").textContent  = t.fps ?? "—";
  $("t-zoom").textContent = t.zoom ? t.zoom.toFixed(2) + "×" : "—";
  $("t-mode").textContent = t.zoom_mode || "—";
  $("t-ptz").textContent  = t.ptz_state || "—";
  $("t-focus").textContent = t.focus != null ? t.focus : "—";
  $("t-user").textContent = t.user_active ? (t.recognised || "you") : "not recognised";

  // Reflect autofocus activity + position, but don't fight the slider while
  // the user is dragging it.
  if ($("focus-state")) $("focus-state").textContent = t.focus_state || "—";
  if (t.focus != null && !focusDragging) {
    $("focus").value = t.focus;
    $("focus-val").textContent = t.focus;
  }

  // Tally: red when a computer is actively connected, green when merely
  // paired-and-ready, grey when nothing is there.
  const tally = $("tally");
  if (tally) {
    tally.dataset.state = t.hid_connected ? "live"
                        : (hasPairedDevice() ? "ready" : "idle");
  }
}

function hasPairedDevice() {
  return document.querySelectorAll("[data-forget]").length > 0;
}

/* ── PTZ mode ────────────────────────────────────────────────────────── */
const manualControls = () => [
  ...document.querySelectorAll(".dpad button"),
  ...document.querySelectorAll(".step-picker button"),
  $("pan"), $("tilt"), $("hwzoom"),
].filter(Boolean);

async function setMode(mode) {
  const data = await post("/api/ptz/mode", { mode });
  const isManual = mode === "manual";

  $("mode-auto").setAttribute("aria-pressed", String(!isManual));
  $("mode-manual").setAttribute("aria-pressed", String(isManual));
  manualControls().forEach((el) => (el.disabled = !isManual));

  $("mode-hint").textContent = isManual
    ? "You're aiming the camera. Switch back to Follow me to let it track you again."
    : "The camera is following the person it recognises. Switch to manual to aim it yourself.";

  paintPTZ(data.status);
}

function paintPTZ(status) {
  if (!status) return;
  const { pan, tilt } = status.position;
  if (pan != null)  { $("pan").value = pan;   $("pan-val").textContent = pan; }
  if (tilt != null) { $("tilt").value = tilt; $("tilt-val").textContent = tilt; }
}

$("mode-auto").onclick   = () => setMode("auto").catch(alertErr);
$("mode-manual").onclick = () => setMode("manual").catch(alertErr);

// D-pad arrows move 1° (the finest step the PTZ board accepts) multiplied
// by the selected step size — 1° for fine aiming, 5°/15° for coarse moves.
let nudgeStep = 1;
document.querySelectorAll(".step-picker button[data-step]").forEach((btn) => {
  btn.onclick = () => {
    nudgeStep = Number(btn.dataset.step);
    document.querySelectorAll(".step-picker button[data-step]")
      .forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
  };
});

document.querySelectorAll(".dpad button[data-pan], .dpad button[data-tilt]")
  .forEach((btn) => {
    btn.onclick = async () => {
      try {
        const data = await post("/api/ptz/nudge", {
          pan:  Number(btn.dataset.pan  || 0) * nudgeStep,
          tilt: Number(btn.dataset.tilt || 0) * nudgeStep,
        });
        paintPTZ(data.status);
      } catch (e) { alertErr(e); }
    };
  });

$("ptz-center").onclick = async () => {
  try { paintPTZ((await post("/api/ptz/center")).status); }
  catch (e) { alertErr(e); }
};

// Sliders: only commit on release, so dragging doesn't flood the motors
// with i2c writes it can't physically keep up with.
["pan", "tilt"].forEach((axis) => {
  const el = $(axis);
  el.oninput  = () => ($(axis + "-val").textContent = el.value);
  el.onchange = () => post("/api/ptz/set", { [axis]: Number(el.value) })
                        .then((d) => paintPTZ(d.status))
                        .catch(alertErr);
});

if ($("hwzoom")) {
  $("hwzoom").oninput  = () => ($("hwzoom-val").textContent = $("hwzoom").value);
  $("hwzoom").onchange = () =>
    post("/api/ptz/set", { hw_zoom: Number($("hwzoom").value) }).catch(alertErr);
}

/* ── Digital auto-zoom ───────────────────────────────────────────────── */
$("digizoom").oninput = () =>
  ($("digizoom-val").textContent = Number($("digizoom").value).toFixed(1) + "×");
$("digizoom").onchange = () =>
  post("/api/zoom", { max_zoom: Number($("digizoom").value) }).catch(alertErr);

$("zoom-enabled").onchange = () =>
  post("/api/zoom", { enabled: $("zoom-enabled").checked }).catch(alertErr);

/* ── Autofocus ───────────────────────────────────────────────────────────── */
let focusDragging = false;

// The manual slider only makes sense when autofocus is off — otherwise the
// controller would immediately hunt away from wherever you set it.
function reflectFocusAuto(isAuto) {
  $("focus-auto").checked = isAuto;
  $("focus").disabled = isAuto;
}

$("focus-auto").onchange = () => {
  const on = $("focus-auto").checked;
  reflectFocusAuto(on);
  post("/api/focus", { auto: on }).catch(alertErr);
};

$("focus-now").onclick = () =>
  post("/api/focus", { refocus: true }).catch(alertErr);

// Dragging turns autofocus off (manual takeover). Commit on release so we
// don't flood the focus motor with i2c writes it can't keep up with.
$("focus").oninput = () => {
  focusDragging = true;
  $("focus-val").textContent = $("focus").value;
};
$("focus").onchange = () => {
  focusDragging = false;
  reflectFocusAuto(false);
  post("/api/focus", { value: Number($("focus").value) }).catch(alertErr);
};

/* ── Pairing ─────────────────────────────────────────────────────────── */
const panes = ["pair-idle", "pair-waiting", "pair-confirm", "pair-error"];
function showPane(id) {
  panes.forEach((p) => ($(p).hidden = p !== id));
}

let pollTimer = null;

// Server-side pairing state stays "paired" indefinitely once ANY device has
// ever paired — it only resets on the next begin_pairing()/cancel_pairing()
// call. So "paired" is not a one-time event we can react to just by seeing
// it; it's only worth reloading for if we actually watched pairing happen
// during this page's lifetime, not merely because that's the leftover state
// from a completed pairing that predates this page load.
let sawInProgress = false;

async function pollPairing() {
  try {
    const data = await (await fetch("/api/bt/status")).json();
    const p = data.pairing;

    if (p.permanent) {
      // Always-discoverable mode has no end state to poll toward — new
      // devices can keep pairing indefinitely, so this pane just stays up
      // until "Stop discovery" is clicked, instead of reloading per pair.
      sawInProgress = true;
      $("pair-waiting-msg").innerHTML = "Always discoverable. Open " +
        "Bluetooth settings on your computer and connect to " +
        "<b>Galaxy Mouse</b> any time.";
      showPane("pair-waiting");
      startPolling();
    } else if (p.state === "confirming" && p.pending) {
      sawInProgress = true;
      $("passkey").textContent = p.pending.passkey || "confirm on device";
      showPane("pair-confirm");
      startPolling();
    } else if (p.state === "pairable") {
      sawInProgress = true;
      $("pair-waiting-msg").innerHTML = "Open Bluetooth settings on your " +
        "computer and connect to <b>Galaxy Mouse</b>.";
      showPane("pair-waiting");
      startPolling();
    } else if (p.state === "paired") {
      stopPolling();
      if (sawInProgress) {
        location.reload();   // simplest way to re-render the device list
      } else {
        showPane("pair-idle");   // stale "paired" from before this page loaded
      }
    } else if (p.state === "failed") {
      stopPolling();
      $("pair-error-msg").textContent = p.error || "Pairing didn't complete.";
      showPane("pair-error");
    } else {
      showPane("pair-idle");
    }
  } catch (e) {
    /* transient — keep polling */
  }
}

function startPolling() {
  if (pollTimer) return;   // already running — avoid stacking intervals
  pollTimer = setInterval(pollPairing, 1000);
}
function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

// Resume the correct pane on page load — e.g. permanent discovery left on
// from an earlier visit, or a 2-minute window still running — rather than
// always defaulting to the idle pane regardless of actual server state.
pollPairing();

$("pair-start").onclick = async () => {
  try {
    await post("/api/bt/pair/start", null, 3);
    await pollPairing();
  } catch (e) {
    $("pair-error-msg").textContent = e.message;
    showPane("pair-error");
  }
};

$("pair-start-permanent").onclick = async () => {
  try {
    await post("/api/bt/pair/start_permanent", null, 3);
    await pollPairing();
  } catch (e) {
    $("pair-error-msg").textContent = e.message;
    showPane("pair-error");
  }
};

$("pair-cancel").onclick = async () => {
  stopPolling();
  await post("/api/bt/pair/cancel").catch(() => {});
  showPane("pair-idle");
};

$("pair-yes").onclick = async () => {
  try {
    await post("/api/bt/pair/confirm", { approve: true }, 5);
  } catch (e) {
    /* 409 = "no pairing request waiting": an earlier attempt whose
       response got lost already confirmed it — let the status polling
       render the real outcome instead of alarming the user. */
    if (e.status !== 409) alertErr(e);
  }
};

$("pair-no").onclick = async () => {
  stopPolling();
  await post("/api/bt/pair/confirm", { approve: false }).catch(() => {});
  showPane("pair-idle");
};

$("pair-retry").onclick = () => showPane("pair-idle");

document.querySelectorAll("[data-forget]").forEach((btn) => {
  btn.onclick = async () => {
    if (!confirm("Forget this computer? You'll need to pair it again.")) return;
    try {
      await post("/api/bt/forget", { mac: btn.dataset.forget });
      location.reload();
    } catch (e) { alertErr(e); }
  };
});

/* ── Face registration ───────────────────────────────────────────────── */
$("face-start").onclick = async () => {
  try {
    await post("/api/face/start");
    $("face-capture").hidden = false;
    $("face-bar").style.width = "0%";
  } catch (e) { alertErr(e); }
};

$("face-cancel").onclick = async () => {
  await post("/api/face/cancel").catch(() => {});
  $("face-capture").hidden = true;
};

if ($("face-clear")) {
  $("face-clear").onclick = async () => {
    if (!confirm("Remove your registered face? Gestures will stop working "
                 + "until you register again.")) return;
    await post("/api/face/clear").catch(alertErr);
    location.reload();
  };
}

socket.on("face_capture_progress", (p) => {
  const pct = Math.round((p.collected / p.needed) * 100);
  $("face-bar").style.width = pct + "%";
  $("face-status").textContent = p.detected
    ? `Hold still — ${p.collected} of ${p.needed}`
    : "Look at the camera. Your face isn't in frame.";
});

socket.on("face_capture_done", () => {
  $("face-status").textContent = "Face registered.";
  setTimeout(() => location.reload(), 800);
});

/* ── Errors ──────────────────────────────────────────────────────────── */
function alertErr(e) {
  alert(e.message || String(e));
}

/* Reflect current PTZ state on load */
fetch("/api/ptz/status")
  .then((r) => r.json())
  .then((d) => paintPTZ(d.ptz))
  .catch(() => {});

/* Reflect current focus state on load */
fetch("/api/focus/status")
  .then((r) => r.json())
  .then((d) => {
    if (d.focus) reflectFocusAuto(d.focus.mode === "auto");
  })
  .catch(() => {});
