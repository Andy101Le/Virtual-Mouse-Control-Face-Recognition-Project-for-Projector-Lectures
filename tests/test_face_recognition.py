"""Offline checks of the new recognition path. No camera, no real faces —
these cover the logic I could have broken, not SFace's own accuracy."""
import os, sys, shutil, sqlite3, tempfile
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import face_recognizer as fr

# Point the module at a scratch copy so the real DB is never touched.
tmp = tempfile.mkdtemp()
shutil.copy(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "login_system.db"), os.path.join(tmp, "t.db"))
fr.DATABASE_NAME = os.path.join(tmp, "t.db")

D = fr.EMBED_DIM


def unit(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=D).astype(np.float32)
    return v / np.linalg.norm(v)


print("== stale (pre-SFace) embeddings must be rejected, not misread ==")
R = fr.FaceRecognizer(active_user="andy101le")
print(f"  usable users : {list(R.face_db)}")
print(f"  need re-enroll: {R.stale_users}")
assert R.face_db == {}, "1434-float geometric rows must not load as SFace"
assert set(R.stale_users) == {"admin", "andy101le", "Machu1287"}
assert R.needs_registration("andy101le")
print("  all three legacy enrollments flagged for re-registration  OK")

print()
print("== DB round-trip: pack -> save -> load -> score ==")
me = unit(1)
samples = [me + 0.05 * unit(100 + i) for i in range(20)]   # 20 noisy views
samples = [s / np.linalg.norm(s) for s in samples]
packed = fr.FaceRecognizer.pack_samples(samples)
print(f"  packed shape {packed.shape} (capped at {fr.MAX_ENROLL_SAMPLES})")
assert packed.shape == (fr.MAX_ENROLL_SAMPLES, D)

conn = sqlite3.connect(fr.DATABASE_NAME)
conn.execute("UPDATE users SET face_embedding=?, face_registered=1 "
             "WHERE username='andy101le'", (packed.astype(np.float32).tobytes(),))
conn.commit(); conn.close()

R = fr.FaceRecognizer(active_user="andy101le")
assert "andy101le" in R.face_db, "round-tripped samples must load"
assert R.face_db["andy101le"].shape == (fr.MAX_ENROLL_SAMPLES, D)
assert not R.needs_registration("andy101le")
print(f"  reloaded {R.face_db['andy101le'].shape}  OK")

print()
print("== matching ==")
genuine = samples[3]
impostor = unit(999)
gs = R.score(genuine, "andy101le")
is_ = R.score(impostor, "andy101le")
print(f"  genuine  score {gs:.3f}   (threshold {fr.RECOGNITION_THRESHOLD})")
print(f"  impostor score {is_:.3f}")
assert gs >= fr.RECOGNITION_THRESHOLD and is_ < fr.RECOGNITION_THRESHOLD

name, score = R.recognise(genuine, face_size=0.20)
assert name == "andy101le", f"genuine should match, got {name}"
name, _ = R.recognise(impostor, face_size=0.20)
assert name == fr.UNKNOWN_LABEL, "impostor must not match"
print("  genuine -> match, impostor -> UNKNOWN                      OK")

print()
print("== distance-aware threshold ramp ==")
for size in (0.30, 0.12, 0.085, 0.05, 0.01):
    print(f"  face_size {size:<5} -> threshold {R._threshold_for_size(size):.3f}")
assert R._threshold_for_size(0.30) == fr.RECOGNITION_THRESHOLD
assert R._threshold_for_size(None) == fr.RECOGNITION_THRESHOLD
floor = fr.RECOGNITION_THRESHOLD - fr.SMALL_FACE_MAX_RELAX
assert abs(R._threshold_for_size(0.01) - floor) < 1e-6
assert R._threshold_for_size(0.085) < R._threshold_for_size(0.12)
print("  monotonic, clamped at both ends                            OK")

print()
print("== REGRESSION: a registered bystander must not steal the lock ==")
# Enrol a second user whose embedding scores HIGHER than the active user's.
bystander = unit(7)
conn = sqlite3.connect(fr.DATABASE_NAME)
conn.execute("UPDATE users SET face_embedding=?, face_registered=1 "
             "WHERE username='admin'",
             (np.stack([bystander]).astype(np.float32).tobytes(),))
conn.commit(); conn.close()

R = fr.FaceRecognizer(active_user="andy101le")
# The live face is the ACTIVE user's, but 'admin' is also in the DB.
name, score = R.recognise(genuine, face_size=0.20)
print(f"  live face = active user, admin also enrolled -> {name} ({score:.3f})")
assert name == "andy101le", "active user must still be recognised"

# And the bystander's own face must never authenticate this session.
name, score = R.recognise(bystander, face_size=0.20)
print(f"  live face = bystander                        -> {name} ({score:.3f})")
assert name == fr.UNKNOWN_LABEL, "bystander must never match in active_user mode"

print()
print("== auth_manager picks the matching face, not face index 0 ==")
import auth_manager as am


class LM:
    def __init__(self, x, y):
        self.x, self.y, self.z = x, y, 0.0


def face(cx, cy):
    # 478 landmarks in a small blob; index 4 is the nose tip.
    return [LM(cx + 0.01 * ((i % 7) - 3), cy + 0.01 * ((i % 5) - 2))
            for i in range(478)]


A = am.AuthManager(active_user="andy101le")
A.face_rec = R
bystander_face = face(0.20, 0.40)      # index 0 = the WRONG person
user_face      = face(0.75, 0.45)      # index 1 = the real user
embeddings = [((0.20, 0.40), bystander), ((0.75, 0.45), genuine)]

A.update([bystander_face, user_face], 1.0, embeddings=embeddings)
print(f"  bystander at index 0, user at index 1 -> recognised '{A.recognised_user}'")
print(f"  tracker anchored at x={A.face_nose_pos[0]:.2f} (user is at 0.75)")
assert A.recognised_user == "andy101le", "must find the user at index 1"
assert A.face_nose_pos[0] > 0.6, "PTZ must track the user, not the bystander"

print()
print("all recognition assertions passed")
shutil.rmtree(tmp)
