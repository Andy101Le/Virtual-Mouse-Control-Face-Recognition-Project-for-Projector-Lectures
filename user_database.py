"""
user_database.py
─────────────────
SQLite storage for accounts, face embeddings, and paired Bluetooth
devices. Pure data-access layer — no Flask, no OpenCV, no BlueZ.

CHANGED FOR THE WEB VERSION
  1. Passwords are now HASHED (werkzeug scrypt), not stored in plain
     text. The Tkinter app stored them as-is, which was survivable when
     the only way in was physical access to the Pi. Exposing a login
     form on the network makes plaintext storage a genuine liability, so
     it's fixed here.

     Existing plaintext rows are migrated on first open — see
     _migrate_plaintext_passwords(). Users keep their current passwords;
     the stored values are just hashed in place.

  2. New `bt_devices` table maps a paired Bluetooth MAC to the account
     that paired it. This is what makes a pairing "belong" to a user.

  3. Thread-safety: the web server touches the DB from request handlers
     AND from the gesture thread, so the connection uses
     check_same_thread=False guarded by a lock.
"""

import sqlite3
import threading
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash


class UserDatabase:
    def __init__(self, db_path):
        self.db_path    = db_path
        self._lock      = threading.RLock()
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.cursor     = self.connection.cursor()
        self._create_tables()
        self._migrate_plaintext_passwords()

    def _create_tables(self):
        with self._lock:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    username        TEXT UNIQUE,
                    password        TEXT,
                    is_admin        INTEGER,
                    face_embedding  BLOB,
                    face_registered INTEGER DEFAULT 0
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS bt_devices (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    mac        TEXT UNIQUE NOT NULL,
                    name       TEXT,
                    username   TEXT NOT NULL,
                    paired_at  TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (username) REFERENCES users(username)
                        ON DELETE CASCADE
                )
            """)
            self.connection.commit()

    def _migrate_plaintext_passwords(self):
        """
        Werkzeug hashes carry a recognisable "method$salt$digest" shape.
        Anything that doesn't is a legacy plaintext password from the
        Tkinter era — hash it in place so the account keeps working.
        """
        with self._lock:
            self.cursor.execute("SELECT username, password FROM users")
            rows = self.cursor.fetchall()
            migrated = 0
            for row in rows:
                pw = row["password"] or ""
                if pw.count("$") >= 2:
                    continue   # already hashed
                self.cursor.execute(
                    "UPDATE users SET password = ? WHERE username = ?",
                    (generate_password_hash(pw), row["username"]))
                migrated += 1
            if migrated:
                self.connection.commit()
                print(f"[UserDatabase] Hashed {migrated} legacy plaintext password(s).")

    # ── Accounts ────────────────────────────────────────────────────────────
    def add_user(self, username, password, is_admin):
        with self._lock:
            try:
                self.cursor.execute("""
                    INSERT INTO users (username, password, is_admin,
                                       face_embedding, face_registered)
                    VALUES (?, ?, ?, NULL, 0)
                """, (username, generate_password_hash(password), int(bool(is_admin))))
                self.connection.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def check_login(self, username, password):
        """Returns the user row on success, None on failure."""
        with self._lock:
            self.cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = self.cursor.fetchone()
        if row and check_password_hash(row["password"], password):
            return row
        return None

    def get_user(self, username):
        with self._lock:
            self.cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            return self.cursor.fetchone()

    def admin_exists(self):
        with self._lock:
            self.cursor.execute("SELECT 1 FROM users WHERE is_admin = 1 LIMIT 1")
            return self.cursor.fetchone() is not None

    def get_all_users(self):
        with self._lock:
            self.cursor.execute("""
                SELECT username, is_admin, face_registered FROM users
                ORDER BY is_admin DESC, username ASC
            """)
            return [dict(r) for r in self.cursor.fetchall()]

    def delete_user(self, username):
        with self._lock:
            self.cursor.execute("DELETE FROM bt_devices WHERE username = ?", (username,))
            self.cursor.execute("DELETE FROM users WHERE username = ?", (username,))
            self.connection.commit()

    def reset_database(self):
        with self._lock:
            self.cursor.execute("DELETE FROM bt_devices")
            self.cursor.execute("DELETE FROM users")
            self.connection.commit()

    def change_username(self, old_username, new_username):
        with self._lock:
            try:
                self.cursor.execute("UPDATE users SET username = ? WHERE username = ?",
                                    (new_username, old_username))
                self.cursor.execute("UPDATE bt_devices SET username = ? WHERE username = ?",
                                    (new_username, old_username))
                self.connection.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def change_password(self, username, new_password):
        with self._lock:
            self.cursor.execute("UPDATE users SET password = ? WHERE username = ?",
                                (generate_password_hash(new_password), username))
            self.connection.commit()

    def set_admin(self, username, is_admin):
        with self._lock:
            self.cursor.execute("UPDATE users SET is_admin = ? WHERE username = ?",
                                (1 if is_admin else 0, username))
            self.connection.commit()

    def count_admins(self):
        with self._lock:
            self.cursor.execute("SELECT COUNT(*) AS n FROM users WHERE is_admin = 1")
            return self.cursor.fetchone()["n"]

    def get_is_admin(self, username):
        row = self.get_user(username)
        return bool(row["is_admin"]) if row else None

    # ── Face embeddings ─────────────────────────────────────────────────────
    def save_face_embedding(self, username, embedding):
        with self._lock:
            self.cursor.execute("""
                UPDATE users SET face_embedding = ?, face_registered = 1
                WHERE username = ?
            """, (embedding.astype(np.float32).tobytes(), username))
            self.connection.commit()

    def clear_face_embedding(self, username):
        with self._lock:
            self.cursor.execute("""
                UPDATE users SET face_embedding = NULL, face_registered = 0
                WHERE username = ?
            """, (username,))
            self.connection.commit()

    def has_face(self, username):
        row = self.get_user(username)
        return bool(row and row["face_registered"])

    # ── Bluetooth device ↔ account mapping ──────────────────────────────────
    def add_bt_device(self, mac, name, username):
        """
        Claims a freshly-paired MAC for a user. Re-pairing an existing MAC
        reassigns it — a physical device can only sensibly belong to one
        account at a time.
        """
        with self._lock:
            self.cursor.execute("""
                INSERT INTO bt_devices (mac, name, username)
                VALUES (?, ?, ?)
                ON CONFLICT(mac) DO UPDATE SET
                    username  = excluded.username,
                    name      = excluded.name,
                    paired_at = CURRENT_TIMESTAMP
            """, (mac.upper(), name, username))
            self.connection.commit()
            return True

    def get_bt_devices(self, username=None):
        with self._lock:
            if username is None:
                self.cursor.execute(
                    "SELECT * FROM bt_devices ORDER BY paired_at DESC")
            else:
                self.cursor.execute(
                    "SELECT * FROM bt_devices WHERE username = ? ORDER BY paired_at DESC",
                    (username,))
            return [dict(r) for r in self.cursor.fetchall()]

    def get_bt_device_owner(self, mac):
        with self._lock:
            self.cursor.execute(
                "SELECT username FROM bt_devices WHERE mac = ?", (mac.upper(),))
            row = self.cursor.fetchone()
            return row["username"] if row else None

    def remove_bt_device(self, mac):
        with self._lock:
            self.cursor.execute("DELETE FROM bt_devices WHERE mac = ?", (mac.upper(),))
            self.connection.commit()
