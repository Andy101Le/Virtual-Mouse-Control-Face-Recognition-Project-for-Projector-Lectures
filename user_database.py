"""
user_database.py
─────────────────
SQLite-backed account + face-embedding storage for the Hand Gesture
Control project. Pure data-access layer — no Tkinter, no OpenCV — so
it stays reusable and testable independent of the GUI.
"""

import sqlite3
import numpy as np


class UserDatabase:
    def __init__(self, db_path):
        self.db_path    = db_path
        self.connection = sqlite3.connect(db_path)
        self.cursor     = self.connection.cursor()
        self._create_tables()

    def _create_tables(self):
        # face_embedding is a BLOB holding a (1434,) float32 numpy array,
        # or NULL if the user has not registered a face yet.
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
        self.connection.commit()

    # ── Account CRUD ───────────────────────────────────────────────────────
    def add_user(self, username, password, is_admin):
        try:
            self.cursor.execute("""
                INSERT INTO users (username, password, is_admin, face_embedding, face_registered)
                VALUES (?, ?, ?, NULL, 0)
            """, (username, password, is_admin))
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def check_login(self, username, password):
        self.cursor.execute("""
            SELECT * FROM users WHERE username = ? AND password = ?
        """, (username, password))
        return self.cursor.fetchone()

    def admin_exists(self):
        self.cursor.execute("SELECT * FROM users WHERE is_admin = 1")
        return self.cursor.fetchone() is not None

    def get_all_users(self):
        self.cursor.execute("SELECT username, is_admin, face_registered FROM users")
        return self.cursor.fetchall()

    def delete_user(self, username):
        self.cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        self.connection.commit()

    def reset_database(self):
        self.cursor.execute("DELETE FROM users")
        self.connection.commit()

    def change_username(self, old_username, new_username):
        """Returns True on success, False if new name already taken."""
        try:
            self.cursor.execute("UPDATE users SET username = ? WHERE username = ?",
                                (new_username, old_username))
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def change_password(self, username, new_password):
        self.cursor.execute("UPDATE users SET password = ? WHERE username = ?",
                            (new_password, username))
        self.connection.commit()

    def set_admin(self, username, is_admin):
        self.cursor.execute("UPDATE users SET is_admin = ? WHERE username = ?",
                            (1 if is_admin else 0, username))
        self.connection.commit()

    def count_admins(self):
        self.cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1")
        return self.cursor.fetchone()[0]

    def get_is_admin(self, username):
        """Returns True/False, or None if the user doesn't exist."""
        self.cursor.execute("SELECT is_admin FROM users WHERE username = ?", (username,))
        row = self.cursor.fetchone()
        return bool(row[0]) if row else None

    # ── Face embeddings ────────────────────────────────────────────────────
    def save_face_embedding(self, username, embedding):
        """Store a numpy embedding for a user. embedding: float32 array."""
        blob = embedding.astype(np.float32).tobytes()
        self.cursor.execute("""
            UPDATE users SET face_embedding = ?, face_registered = 1
            WHERE username = ?
        """, (blob, username))
        self.connection.commit()

    def clear_face_embedding(self, username):
        self.cursor.execute("""
            UPDATE users SET face_embedding = NULL, face_registered = 0
            WHERE username = ?
        """, (username,))
        self.connection.commit()

    def has_face(self, username):
        self.cursor.execute("SELECT face_registered FROM users WHERE username = ?", (username,))
        row = self.cursor.fetchone()
        return bool(row and row[0])
