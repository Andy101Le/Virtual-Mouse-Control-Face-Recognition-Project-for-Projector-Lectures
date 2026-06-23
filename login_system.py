"""
login_system.py
───────────────
Account + face database for the Hand Gesture Control project.

Same Tkinter login flow as before, extended with:
  - Face embeddings stored in SQLite (no more faces.pkl)
  - "Register My Face" for the logged-in user
  - Admin tools to enroll / clear faces for other users
  - "Launch Gesture Control" runs main.py with the current user

Run this FIRST. main.py is launched from inside the main menu.
"""

import sqlite3
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
import subprocess
import sys
import os
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


# =========================================================
# CONFIG
# =========================================================

# Resolve paths relative to this script's folder, not the cwd PyCharm uses
_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATABASE_NAME   = os.path.join(_SCRIPT_DIR, "login_system.db")
FACE_TASK_PATH  = os.path.join(_SCRIPT_DIR, "face_landmarker.task")
SAMPLES_NEEDED  = 60
CAPTURE_DELAY   = 0.05


# =========================================================
# CONNECT TO DATABASE
# =========================================================

connection = sqlite3.connect(DATABASE_NAME)
cursor     = connection.cursor()


# =========================================================
# CREATE USERS TABLE
# =========================================================
# face_embedding is a BLOB holding a (1434,) float32 numpy array,
# or NULL if the user has not registered a face yet.

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT UNIQUE,
    password        TEXT,
    is_admin        INTEGER,
    face_embedding  BLOB,
    face_registered INTEGER DEFAULT 0
)
""")
connection.commit()


# =========================================================
# GLOBAL VARIABLES
# =========================================================

current_user          = ""
current_user_is_admin = False


# =========================================================
# DATABASE FUNCTIONS
# =========================================================

def add_user(username, password, is_admin):
    try:
        cursor.execute("""
            INSERT INTO users (username, password, is_admin, face_embedding, face_registered)
            VALUES (?, ?, ?, NULL, 0)
        """, (username, password, is_admin))
        connection.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def check_login(username, password):
    cursor.execute("""
        SELECT * FROM users
        WHERE username = ? AND password = ?
    """, (username, password))
    return cursor.fetchone()


def admin_exists():
    cursor.execute("SELECT * FROM users WHERE is_admin = 1")
    return cursor.fetchone() is not None


def get_all_users():
    cursor.execute("SELECT username, is_admin, face_registered FROM users")
    return cursor.fetchall()


def delete_user(username):
    cursor.execute("DELETE FROM users WHERE username = ?", (username,))
    connection.commit()


def reset_database():
    cursor.execute("DELETE FROM users")
    connection.commit()


def save_face_embedding(username, embedding):
    """Store a numpy embedding for a user. embedding: float32 array."""
    blob = embedding.astype(np.float32).tobytes()
    cursor.execute("""
        UPDATE users
        SET face_embedding = ?, face_registered = 1
        WHERE username = ?
    """, (blob, username))
    connection.commit()


def clear_face_embedding(username):
    cursor.execute("""
        UPDATE users
        SET face_embedding = NULL, face_registered = 0
        WHERE username = ?
    """, (username,))
    connection.commit()


def has_face(username):
    cursor.execute("SELECT face_registered FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    return bool(row and row[0])


def change_username(old_username, new_username):
    """Returns True on success, False if new name already taken or other error."""
    try:
        cursor.execute("UPDATE users SET username = ? WHERE username = ?",
                       (new_username, old_username))
        connection.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def change_password(username, new_password):
    cursor.execute("UPDATE users SET password = ? WHERE username = ?",
                   (new_password, username))
    connection.commit()


def set_admin(username, is_admin):
    cursor.execute("UPDATE users SET is_admin = ? WHERE username = ?",
                   (1 if is_admin else 0, username))
    connection.commit()


def count_admins():
    cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1")
    return cursor.fetchone()[0]


# =========================================================
# FACE CAPTURE (OpenCV window — used by both regular users
# enrolling themselves and admins enrolling others)
# =========================================================

def extract_embedding(face_landmarks):
    """Same normalisation as the rest of the project — must match exactly."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32)
    pts -= pts[4]                            # centre on nose tip
    scale = np.max(np.abs(pts)) or 1.0
    pts  /= scale
    flat  = pts.flatten()
    norm  = np.linalg.norm(flat) or 1.0
    return flat / norm                       # L2-normalised for cosine similarity


def capture_face_for_user(target_username):
    """
    Open an OpenCV window, capture SAMPLES_NEEDED face samples,
    average them, and save to the DB for target_username.
    Returns True on success, False if cancelled / no samples.
    """
    if not os.path.exists(FACE_TASK_PATH):
        messagebox.showerror(
            "Missing model",
            f"'{FACE_TASK_PATH}' not found. Run download_models.py first."
        )
        return False

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_TASK_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    start_time        = time.perf_counter()
    sample_embeddings = []
    capturing         = False
    last_capture_t    = 0.0
    saved             = False

    print(f"\n=== Capturing face for '{target_username}' ===")
    print("Hold SPACE to capture 60 frames.  Q to cancel.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame  = np.ascontiguousarray(frame[:, ::-1, :])    # mirror
        h, w   = frame.shape[:2]
        rgb    = np.ascontiguousarray(frame[:, :, ::-1])
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int((time.perf_counter() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        face_detected = bool(result.face_landmarks)
        n_collected   = len(sample_embeddings)
        done          = n_collected >= SAMPLES_NEEDED

        # HUD
        cv2.rectangle(frame, (0, 0), (w, 75), (20, 20, 20), -1)
        cv2.putText(frame, f"User: {target_username}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if done:
            status_txt = "DONE - press Q to save"
        elif capturing:
            status_txt = f"CAPTURED {n_collected}/{SAMPLES_NEEDED}"
        else:
            status_txt = "Hold SPACE to capture"
        status_col = (0, 220, 0) if face_detected else (0, 0, 220)
        cv2.putText(frame, status_txt, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

        # Progress bar
        bar_w = int((n_collected / SAMPLES_NEEDED) * w)
        cv2.rectangle(frame, (0, h - 10), (bar_w, h), (0, 200, 80), -1)

        # Draw a few key landmarks + face box
        if result.face_landmarks:
            face = result.face_landmarks[0]
            for idx in [4, 33, 263, 61, 291, 199]:
                lm = face[idx]
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0, 255, 0), -1)
            xs = [int(lm.x * w) for lm in face]
            ys = [int(lm.y * h) for lm in face]
            x1, x2 = max(min(xs)-10, 0), min(max(xs)+10, w)
            y1, y2 = max(min(ys)-10, 0), min(max(ys)+10, h)
            box_col = (0, 255, 0) if (capturing and not done) else (200, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2)

        cv2.imshow(f"Face Registration - {target_username}", frame)
        key = cv2.waitKey(1) & 0xFF
        now = time.perf_counter()

        if key == 32 and face_detected and not done:
            capturing = True

        if capturing and face_detected and not done and (now - last_capture_t) >= CAPTURE_DELAY:
            emb = extract_embedding(result.face_landmarks[0])
            sample_embeddings.append(emb)
            last_capture_t = now
            if len(sample_embeddings) >= SAMPLES_NEEDED:
                capturing = False
                print(f"  Captured {SAMPLES_NEEDED} samples - press Q to save.")

        if key == ord('q') or key == 27:
            break

    if sample_embeddings:
        # Average + L2-normalise for cosine similarity
        mean_emb  = np.mean(sample_embeddings, axis=0)
        mean_emb /= (np.linalg.norm(mean_emb) or 1.0)
        save_face_embedding(target_username, mean_emb)
        saved = True
        print(f"Saved face for '{target_username}'.")
    else:
        print("No samples collected - nothing saved.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    return saved


# =========================================================
# TKINTER WINDOW
# =========================================================

root = tk.Tk()
root.title("Hand Gesture Control - Login")
root.geometry("500x600")
root.configure(bg="#202020")


def clear_window():
    for widget in root.winfo_children():
        widget.destroy()


def create_title(text):
    label = tk.Label(root, text=text, font=("Arial", 22, "bold"),
                     bg="#202020", fg="white")
    label.pack(pady=20)


def create_button(text, command):
    button = tk.Button(root, text=text, command=command,
                       width=30, height=2, font=("Arial", 12))
    button.pack(pady=8)


# =========================================================
# FIRST TIME ADMIN SETUP
# =========================================================

def show_create_admin_screen():
    clear_window()
    create_title("Create Admin Account")

    tk.Label(root, text="Username", bg="#202020", fg="white").pack()
    username_entry = tk.Entry(root, width=30)
    username_entry.pack(pady=5)

    tk.Label(root, text="Password", bg="#202020", fg="white").pack()
    password_entry = tk.Entry(root, width=30, show="*")
    password_entry.pack(pady=5)

    def create_admin():
        username = username_entry.get()
        password = password_entry.get()
        if username == "" or password == "":
            messagebox.showerror("Error", "Fill in all fields")
            return
        if add_user(username, password, 1):
            messagebox.showinfo("Success", "Admin account created")
            show_login_screen()
        else:
            messagebox.showerror("Error", "Username already exists")

    create_button("Create Admin Account", create_admin)
    create_button("Quit Program", root.destroy)


# =========================================================
# CREATE ACCOUNT SCREEN
# =========================================================

def show_create_account_screen():
    clear_window()
    create_title("Create User Account")

    tk.Label(root, text="Username", bg="#202020", fg="white").pack()
    username_entry = tk.Entry(root, width=30)
    username_entry.pack(pady=5)

    tk.Label(root, text="Password", bg="#202020", fg="white").pack()
    password_entry = tk.Entry(root, width=30, show="*")
    password_entry.pack(pady=5)

    def create_account():
        username = username_entry.get()
        password = password_entry.get()
        if username == "" or password == "":
            messagebox.showerror("Error", "Fill in all fields")
            return
        if add_user(username, password, 0):
            messagebox.showinfo("Success", "Account created. Please log in to register your face.")
            show_login_screen()
        else:
            messagebox.showerror("Error", "Username already exists")

    create_button("Create Account", create_account)
    create_button("Back", show_login_screen)


# =========================================================
# LOGIN SCREEN
# =========================================================

def show_login_screen():
    clear_window()
    create_title("Login")

    tk.Label(root, text="Username", bg="#202020", fg="white").pack()
    username_entry = tk.Entry(root, width=30)
    username_entry.pack(pady=5)

    tk.Label(root, text="Password", bg="#202020", fg="white").pack()
    password_entry = tk.Entry(root, width=30, show="*")
    password_entry.pack(pady=5)

    def login():
        global current_user, current_user_is_admin
        username = username_entry.get()
        password = password_entry.get()
        user = check_login(username, password)
        if user:
            current_user          = username
            current_user_is_admin = (user[3] == 1)
            messagebox.showinfo("Success", "Login successful")
            show_main_menu()
        else:
            messagebox.showerror("Error", "Wrong username or password")

    create_button("Login", login)
    create_button("Create Account", show_create_account_screen)
    create_button("Quit Program", root.destroy)


# =========================================================
# SHOW USERS
# =========================================================

def show_user_list():
    users = get_all_users()
    text = ""
    for username, is_admin, face_reg in users:
        role     = "Admin" if is_admin == 1 else "User"
        face_str = "face OK" if face_reg else "no face"
        text += f"{username} - {role} - {face_str}\n"
    messagebox.showinfo("Users", text or "No users")


# =========================================================
# REGISTER MY FACE
# =========================================================

def register_my_face():
    if has_face(current_user):
        answer = messagebox.askyesno(
            "Face already registered",
            "You already have a face on file. Replace it?"
        )
        if not answer:
            return
    capture_face_for_user(current_user)
    show_main_menu()  # refresh menu


# =========================================================
# DATABASE MANAGER (admin only)
# =========================================================

def open_database_manager():
    if not current_user_is_admin:
        messagebox.showerror("Error", "Admins only")
        return

    manager = tk.Toplevel(root)
    manager.title("Database Manager")
    manager.geometry("560x680")
    manager.configure(bg="#202020")

    tk.Label(manager, text="Database Manager",
             font=("Arial", 20, "bold"),
             bg="#202020", fg="white").pack(pady=15)

    tk.Label(manager,
             text="Select a user, then choose an action",
             bg="#202020", fg="#aaaaaa",
             font=("Arial", 10)).pack()

    user_list = tk.Listbox(manager, width=55, height=12,
                           font=("Consolas", 10))
    user_list.pack(pady=10)

    for username, is_admin, face_reg in get_all_users():
        role     = "Admin" if is_admin == 1 else "User "
        face_str = "face OK" if face_reg else "no face"
        user_list.insert(tk.END, f"{username:<20} {role}  {face_str}")

    def selected_username():
        sel = user_list.curselection()
        if not sel:
            messagebox.showinfo("No selection", "Pick a user from the list first.")
            return None
        return user_list.get(sel).split()[0]

    def refresh():
        manager.destroy()
        open_database_manager()

    # ── Account actions ──────────────────────────────────────────────────────
    def add_new_user():
        uname = simpledialog.askstring("Add User", "New username:", parent=manager)
        if not uname:
            return
        pword = simpledialog.askstring("Add User", f"Password for {uname}:",
                                       show="*", parent=manager)
        if not pword:
            return
        make_admin = messagebox.askyesno("Add User", f"Make {uname} an admin?",
                                          parent=manager)
        if add_user(uname, pword, 1 if make_admin else 0):
            messagebox.showinfo("Success", f"User '{uname}' created", parent=manager)
            refresh()
        else:
            messagebox.showerror("Error", "Username already exists", parent=manager)

    def delete_selected_user():
        username = selected_username()
        if not username:
            return
        if username == current_user:
            messagebox.showerror("Error", "Cannot delete the currently logged-in user",
                                 parent=manager)
            return
        # Prevent deleting the last admin
        cursor.execute("SELECT is_admin FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row and row[0] == 1 and count_admins() <= 1:
            messagebox.showerror("Error", "Cannot delete the last admin account",
                                 parent=manager)
            return
        if not messagebox.askyesno("Confirm", f"Delete user '{username}'?",
                                    parent=manager):
            return
        delete_user(username)
        messagebox.showinfo("Success", "User deleted", parent=manager)
        refresh()

    def rename_selected_user():
        global current_user
        username = selected_username()
        if not username:
            return
        new_name = simpledialog.askstring("Rename User",
                                          f"New username for '{username}':",
                                          parent=manager)
        if not new_name or new_name == username:
            return
        if change_username(username, new_name):
            # If we renamed the currently-logged-in user, update the global
            if username == current_user:
                current_user = new_name
            messagebox.showinfo("Success", f"Renamed to '{new_name}'", parent=manager)
            refresh()
        else:
            messagebox.showerror("Error", "Username already taken", parent=manager)

    def change_selected_password():
        username = selected_username()
        if not username:
            return
        new_pw = simpledialog.askstring("Change Password",
                                        f"New password for '{username}':",
                                        show="*", parent=manager)
        if not new_pw:
            return
        confirm = simpledialog.askstring("Change Password",
                                         "Confirm new password:",
                                         show="*", parent=manager)
        if new_pw != confirm:
            messagebox.showerror("Error", "Passwords do not match", parent=manager)
            return
        change_password(username, new_pw)
        messagebox.showinfo("Success", "Password updated", parent=manager)

    def toggle_admin_status():
        username = selected_username()
        if not username:
            return
        cursor.execute("SELECT is_admin FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if not row:
            return
        currently_admin = (row[0] == 1)

        # Prevent demoting the last admin
        if currently_admin and count_admins() <= 1:
            messagebox.showerror("Error", "Cannot demote the last admin", parent=manager)
            return
        # Prevent demoting yourself (would lose access to this window)
        if currently_admin and username == current_user:
            messagebox.showerror("Error",
                                 "Cannot demote yourself while logged in",
                                 parent=manager)
            return

        new_role = "User" if currently_admin else "Admin"
        if not messagebox.askyesno("Confirm",
                                    f"Change '{username}' to {new_role}?",
                                    parent=manager):
            return
        set_admin(username, not currently_admin)
        messagebox.showinfo("Success", f"'{username}' is now {new_role}", parent=manager)
        refresh()

    # ── Face actions (kept from before) ──────────────────────────────────────
    def enroll_selected_face():
        username = selected_username()
        if not username:
            return
        manager.withdraw()
        root.withdraw()
        capture_face_for_user(username)
        root.deiconify()
        refresh()

    def clear_selected_face():
        username = selected_username()
        if not username:
            return
        clear_face_embedding(username)
        messagebox.showinfo("Success", f"Face cleared for {username}", parent=manager)
        refresh()

    # ── Two-column button layout ─────────────────────────────────────────────
    btn_frame = tk.Frame(manager, bg="#202020")
    btn_frame.pack(pady=10)

    BTN_W, BTN_H = 22, 2

    # Left column: account
    left = tk.Frame(btn_frame, bg="#202020")
    left.grid(row=0, column=0, padx=8, sticky="n")
    tk.Label(left, text="Account",
             bg="#202020", fg="#cccccc",
             font=("Arial", 11, "bold")).pack(pady=(0, 5))
    tk.Button(left, text="Add User",         command=add_new_user,
              width=BTN_W, height=BTN_H).pack(pady=3)
    tk.Button(left, text="Delete User",      command=delete_selected_user,
              width=BTN_W, height=BTN_H).pack(pady=3)
    tk.Button(left, text="Change Username",  command=rename_selected_user,
              width=BTN_W, height=BTN_H).pack(pady=3)
    tk.Button(left, text="Change Password",  command=change_selected_password,
              width=BTN_W, height=BTN_H).pack(pady=3)
    tk.Button(left, text="Toggle Admin",     command=toggle_admin_status,
              width=BTN_W, height=BTN_H).pack(pady=3)

    # Right column: face
    right = tk.Frame(btn_frame, bg="#202020")
    right.grid(row=0, column=1, padx=8, sticky="n")
    tk.Label(right, text="Face",
             bg="#202020", fg="#cccccc",
             font=("Arial", 11, "bold")).pack(pady=(0, 5))
    tk.Button(right, text="Register Face",   command=enroll_selected_face,
              width=BTN_W, height=BTN_H).pack(pady=3)
    tk.Button(right, text="Clear Face",      command=clear_selected_face,
              width=BTN_W, height=BTN_H).pack(pady=3)

    tk.Button(manager, text="Close", command=manager.destroy,
              width=20, height=1).pack(pady=10)


# =========================================================
# LAUNCH GESTURE CONTROL
# =========================================================

def launch_gesture_control():
    if not has_face(current_user):
        messagebox.showerror(
            "Face not registered",
            "You must register your face before launching gesture control."
        )
        return

    # Hide the Tk window while main.py owns the camera + display
    root.withdraw()
    try:
        main_path = os.path.join(_SCRIPT_DIR, "main.py")
        subprocess.run([sys.executable, main_path, "--user", current_user],
                       cwd=_SCRIPT_DIR)
    except FileNotFoundError:
        messagebox.showerror("Error", "main.py not found")
    root.deiconify()


# =========================================================
# MAIN MENU
# =========================================================

def show_main_menu():
    clear_window()
    face_str = "(face registered)" if has_face(current_user) else "(no face yet)"
    create_title(f"Welcome {current_user}")
    tk.Label(root, text=face_str, bg="#202020", fg="#aaaaaa",
             font=("Arial", 11)).pack()

    create_button("Launch Gesture Control",  launch_gesture_control)
    create_button("Register My Face",        register_my_face)
    create_button("See User List",           show_user_list)

    if current_user_is_admin:
        create_button("Open Database Manager", open_database_manager)
        create_button("Reset Everything",      reset_everything)

    create_button("Logout", show_login_screen)


# =========================================================
# RESET EVERYTHING
# =========================================================

def reset_everything():
    if not messagebox.askyesno("Warning", "Delete EVERYTHING?"):
        return
    reset_database()
    messagebox.showinfo("Reset", "Everything deleted")
    show_create_admin_screen()


# =========================================================
# START PROGRAM
# =========================================================

if admin_exists():
    show_login_screen()
else:
    show_create_admin_screen()

root.mainloop()