"""
login_app.py
─────────────
Tkinter GUI for the Hand Gesture Control project's account system:
admin setup, login/create-account screens, the main menu, and the
admin database manager.

This class owns widgets and screen flow ONLY — persistence goes through
UserDatabase, face capture goes through FaceCapture, and launching the
gesture-control loop goes through GestureLauncher. None of those three
know Tkinter exists.
"""

import tkinter as tk
from tkinter import messagebox, simpledialog


class LoginApp:
    def __init__(self, db, face_capture, gesture_launcher):
        self.db               = db
        self.face_capture     = face_capture
        self.gesture_launcher = gesture_launcher

        self.current_user          = ""
        self.current_user_is_admin = False

        self.root = tk.Tk()
        self.root.title("Hand Gesture Control - Login")
        self.root.geometry("500x600")
        self.root.configure(bg="#202020")

    def run(self):
        if self.db.admin_exists():
            self.show_login_screen()
        else:
            self.show_create_admin_screen()
        self.root.mainloop()

    # ── Screen-building helpers ─────────────────────────────────────────────
    def _clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def _create_title(self, text):
        tk.Label(self.root, text=text, font=("Arial", 22, "bold"),
                 bg="#202020", fg="white").pack(pady=20)

    def _create_button(self, text, command):
        tk.Button(self.root, text=text, command=command,
                  width=30, height=2, font=("Arial", 12)).pack(pady=8)

    # ── First-time admin setup ──────────────────────────────────────────────
    def show_create_admin_screen(self):
        self._clear_window()
        self._create_title("Create Admin Account")

        tk.Label(self.root, text="Username", bg="#202020", fg="white").pack()
        username_entry = tk.Entry(self.root, width=30)
        username_entry.pack(pady=5)

        tk.Label(self.root, text="Password", bg="#202020", fg="white").pack()
        password_entry = tk.Entry(self.root, width=30, show="*")
        password_entry.pack(pady=5)

        def create_admin():
            username = username_entry.get()
            password = password_entry.get()
            if username == "" or password == "":
                messagebox.showerror("Error", "Fill in all fields")
                return
            if self.db.add_user(username, password, 1):
                messagebox.showinfo("Success", "Admin account created")
                self.show_login_screen()
            else:
                messagebox.showerror("Error", "Username already exists")

        self._create_button("Create Admin Account", create_admin)
        self._create_button("Quit Program", self.root.destroy)

    # ── Create account ───────────────────────────────────────────────────────
    def show_create_account_screen(self):
        self._clear_window()
        self._create_title("Create User Account")

        tk.Label(self.root, text="Username", bg="#202020", fg="white").pack()
        username_entry = tk.Entry(self.root, width=30)
        username_entry.pack(pady=5)

        tk.Label(self.root, text="Password", bg="#202020", fg="white").pack()
        password_entry = tk.Entry(self.root, width=30, show="*")
        password_entry.pack(pady=5)

        def create_account():
            username = username_entry.get()
            password = password_entry.get()
            if username == "" or password == "":
                messagebox.showerror("Error", "Fill in all fields")
                return
            if self.db.add_user(username, password, 0):
                messagebox.showinfo("Success", "Account created. Please log in to register your face.")
                self.show_login_screen()
            else:
                messagebox.showerror("Error", "Username already exists")

        self._create_button("Create Account", create_account)
        self._create_button("Back", self.show_login_screen)

    # ── Login ────────────────────────────────────────────────────────────────
    def show_login_screen(self):
        self._clear_window()
        self._create_title("Login")

        tk.Label(self.root, text="Username", bg="#202020", fg="white").pack()
        username_entry = tk.Entry(self.root, width=30)
        username_entry.pack(pady=5)

        tk.Label(self.root, text="Password", bg="#202020", fg="white").pack()
        password_entry = tk.Entry(self.root, width=30, show="*")
        password_entry.pack(pady=5)

        def login():
            username = username_entry.get()
            password = password_entry.get()
            user = self.db.check_login(username, password)
            if user:
                self.current_user          = username
                self.current_user_is_admin = (user[3] == 1)
                messagebox.showinfo("Success", "Login successful")
                self.show_main_menu()
            else:
                messagebox.showerror("Error", "Wrong username or password")

        self._create_button("Login", login)
        self._create_button("Create Account", self.show_create_account_screen)
        self._create_button("Quit Program", self.root.destroy)

    # ── User list ─────────────────────────────────────────────────────────────
    def show_user_list(self):
        users = self.db.get_all_users()
        text = ""
        for username, is_admin, face_reg in users:
            role     = "Admin" if is_admin == 1 else "User"
            face_str = "face OK" if face_reg else "no face"
            text += f"{username} - {role} - {face_str}\n"
        messagebox.showinfo("Users", text or "No users")

    # ── Register my face ──────────────────────────────────────────────────────
    def register_my_face(self):
        if self.db.has_face(self.current_user):
            if not messagebox.askyesno(
                "Face already registered",
                "You already have a face on file. Replace it?"
            ):
                return
        embedding = self.face_capture.capture_for_user(self.current_user)
        if embedding is not None:
            self.db.save_face_embedding(self.current_user, embedding)
        self.show_main_menu()  # refresh menu

    # ── Database manager (admin only) ─────────────────────────────────────────
    def open_database_manager(self):
        if not self.current_user_is_admin:
            messagebox.showerror("Error", "Admins only")
            return

        manager = tk.Toplevel(self.root)
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

        for username, is_admin, face_reg in self.db.get_all_users():
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
            self.open_database_manager()

        # ── Account actions ────────────────────────────────────────────────
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
            if self.db.add_user(uname, pword, 1 if make_admin else 0):
                messagebox.showinfo("Success", f"User '{uname}' created", parent=manager)
                refresh()
            else:
                messagebox.showerror("Error", "Username already exists", parent=manager)

        def delete_selected_user():
            username = selected_username()
            if not username:
                return
            if username == self.current_user:
                messagebox.showerror("Error", "Cannot delete the currently logged-in user",
                                     parent=manager)
                return
            # Prevent deleting the last admin
            if self.db.get_is_admin(username) and self.db.count_admins() <= 1:
                messagebox.showerror("Error", "Cannot delete the last admin account",
                                     parent=manager)
                return
            if not messagebox.askyesno("Confirm", f"Delete user '{username}'?",
                                        parent=manager):
                return
            self.db.delete_user(username)
            messagebox.showinfo("Success", "User deleted", parent=manager)
            refresh()

        def rename_selected_user():
            username = selected_username()
            if not username:
                return
            new_name = simpledialog.askstring("Rename User",
                                              f"New username for '{username}':",
                                              parent=manager)
            if not new_name or new_name == username:
                return
            if self.db.change_username(username, new_name):
                # If we renamed the currently-logged-in user, update our copy
                if username == self.current_user:
                    self.current_user = new_name
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
            self.db.change_password(username, new_pw)
            messagebox.showinfo("Success", "Password updated", parent=manager)

        def toggle_admin_status():
            username = selected_username()
            if not username:
                return
            currently_admin = self.db.get_is_admin(username)
            if currently_admin is None:
                return

            # Prevent demoting the last admin
            if currently_admin and self.db.count_admins() <= 1:
                messagebox.showerror("Error", "Cannot demote the last admin", parent=manager)
                return
            # Prevent demoting yourself (would lose access to this window)
            if currently_admin and username == self.current_user:
                messagebox.showerror("Error",
                                     "Cannot demote yourself while logged in",
                                     parent=manager)
                return

            new_role = "User" if currently_admin else "Admin"
            if not messagebox.askyesno("Confirm",
                                        f"Change '{username}' to {new_role}?",
                                        parent=manager):
                return
            self.db.set_admin(username, not currently_admin)
            messagebox.showinfo("Success", f"'{username}' is now {new_role}", parent=manager)
            refresh()

        # ── Face actions ───────────────────────────────────────────────────
        def enroll_selected_face():
            username = selected_username()
            if not username:
                return
            manager.withdraw()
            self.root.withdraw()
            embedding = self.face_capture.capture_for_user(username)
            if embedding is not None:
                self.db.save_face_embedding(username, embedding)
            self.root.deiconify()
            refresh()

        def clear_selected_face():
            username = selected_username()
            if not username:
                return
            self.db.clear_face_embedding(username)
            messagebox.showinfo("Success", f"Face cleared for {username}", parent=manager)
            refresh()

        # ── Two-column button layout ─────────────────────────────────────────
        btn_frame = tk.Frame(manager, bg="#202020")
        btn_frame.pack(pady=10)

        BTN_W, BTN_H = 22, 2

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

    # ── Launch gesture control ───────────────────────────────────────────────
    def launch_gesture_control(self):
        if not self.db.has_face(self.current_user):
            messagebox.showerror(
                "Face not registered",
                "You must register your face before launching gesture control."
            )
            return

        # Hide the Tk window while main.py owns the camera + display
        self.root.withdraw()
        if not self.gesture_launcher.launch(self.current_user):
            messagebox.showerror("Error", "main.py not found")
        self.root.deiconify()

    # ── Main menu ──────────────────────────────────────────────────────────────
    def show_main_menu(self):
        self._clear_window()
        face_str = "(face registered)" if self.db.has_face(self.current_user) else "(no face yet)"
        self._create_title(f"Welcome {self.current_user}")
        tk.Label(self.root, text=face_str, bg="#202020", fg="#aaaaaa",
                 font=("Arial", 11)).pack()

        self._create_button("Launch Gesture Control",  self.launch_gesture_control)
        self._create_button("Register My Face",        self.register_my_face)
        self._create_button("See User List",           self.show_user_list)

        if self.current_user_is_admin:
            self._create_button("Open Database Manager", self.open_database_manager)
            self._create_button("Reset Everything",      self.reset_everything)

        self._create_button("Logout", self.show_login_screen)

    # ── Reset everything ──────────────────────────────────────────────────────
    def reset_everything(self):
        if not messagebox.askyesno("Warning", "Delete EVERYTHING?"):
            return
        self.db.reset_database()
        messagebox.showinfo("Reset", "Everything deleted")
        self.show_create_admin_screen()
