/* admin.js — account management, device removal, reset */

const $ = (id) => document.getElementById(id);

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function fail(e) {
  alert(e.message || String(e));
}

/* ── Per-row account actions ─────────────────────────────────────────── */
document.querySelectorAll("tr[data-user] button[data-act]").forEach((btn) => {
  const username = btn.closest("tr").dataset.user;

  btn.onclick = async () => {
    try {
      switch (btn.dataset.act) {
        case "rename": {
          const next = prompt(`New username for "${username}"`, username);
          if (!next || next === username) return;
          await post("/api/admin/user/rename",
                     { username, new_username: next });
          break;
        }

        case "password": {
          const pw = prompt(`New password for "${username}"`);
          if (!pw) return;
          await post("/api/admin/user/password", { username, password: pw });
          alert(`Password changed for ${username}.`);
          return;   // nothing on the page changed
        }

        case "toggle_admin":
          await post("/api/admin/user/toggle_admin", { username });
          break;

        case "clear_face":
          if (!confirm(`Clear the registered face for "${username}"? `
                     + `They'll need to register again before gestures work.`)) return;
          await post("/api/admin/user/clear_face", { username });
          break;

        case "delete":
          if (!confirm(`Delete "${username}"? This also removes their paired `
                     + `computers and registered face.`)) return;
          await post("/api/admin/user/delete", { username });
          break;

        default:
          return;
      }
      location.reload();
    } catch (e) { fail(e); }
  };
});

/* ── Add an account ──────────────────────────────────────────────────── */
$("add-user").onclick = async () => {
  const username = $("new-username").value.trim();
  const password = $("new-password").value;
  if (!username || !password) {
    alert("Enter both a username and a password.");
    return;
  }
  try {
    await post("/api/admin/user/add", {
      username,
      password,
      is_admin: $("new-admin").checked,
    });
    location.reload();
  } catch (e) { fail(e); }
};

/* ── Forget a paired computer ────────────────────────────────────────── */
document.querySelectorAll("[data-forget]").forEach((btn) => {
  btn.onclick = async () => {
    if (!confirm("Forget this computer? Its owner will need to pair it again.")) return;
    try {
      await post("/api/bt/forget", { mac: btn.dataset.forget });
      location.reload();
    } catch (e) { fail(e); }
  };
});

/* ── Danger zone ─────────────────────────────────────────────────────── */
$("reset-all").onclick = async () => {
  if (!confirm("Erase every account, face, and pairing on this Pi?")) return;
  if (!confirm("This can't be undone. Erase everything?")) return;
  try {
    await post("/api/admin/reset");
    location.href = "/signup";
  } catch (e) { fail(e); }
};
