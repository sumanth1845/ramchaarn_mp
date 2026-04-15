"""
train_model.py — Train the GRU cognitive load classifier locally
================================================================
Only needs: session_features.csv (in the same folder)
Produces:   final_model.keras + session_scaler.pkl

Usage:
    python train_model.py
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

# FIX: use "import keras" directly (tensorflow.keras fails on some Windows installs)
import keras
from keras import Model
from keras.layers import Input, GRU, Dense, Dropout, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

print(f"TensorFlow {tf.__version__}  |  Keras {keras.__version__}")
print("-" * 50)

# ═══════════════════════════════════════════
# CONFIG — change these if needed
# ═══════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "session_features.csv")
OUT_MODEL = os.path.join(BASE_DIR, "final_model.keras")
OUT_SCALER = os.path.join(BASE_DIR, "session_scaler.pkl")

SEQ_LEN = 4
RANDOM_STATE = 42

FEATURE_COLS = [
    "avg_dwell", "avg_flight", "typing_speed",
    "pause_count", "backspace_count",
    "std_dwell", "std_flight",
]

# ═══════════════════════════════════════════
# 1. LOAD & CLEAN DATA
# ═══════════════════════════════════════════

print(f"\n[1/7] Loading data from {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    print(f"\nFile not found: {CSV_PATH}")
    print("   Place session_features.csv in the same folder as this script.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"      Shape: {df.shape}")

# Clean
df.columns = [str(c).strip() for c in df.columns]
required = ["user_id", "session_id", "window_id", "timestep"] + FEATURE_COLS
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"\nMissing columns: {missing}")
    sys.exit(1)

for c in ["window_id", "timestep"] + FEATURE_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["std_flight"] = df["std_flight"].fillna(0.0)
for c in FEATURE_COLS:
    df[c] = df[c].fillna(df[c].median())

df = df.drop_duplicates().copy()
df = df.sort_values(["user_id", "session_id", "window_id"], kind="mergesort").reset_index(drop=True)

print(f"      After cleaning: {df.shape}")
print(f"      Users: {df['user_id'].nunique()}, Sessions: {df['session_id'].nunique()}")

# ═══════════════════════════════════════════
# 2. TRAIN / VAL / TEST SPLIT (by user)
# ═══════════════════════════════════════════

print("\n[2/7] Splitting by user (70% / 15% / 15%)")

def split_by_user(frame):
    groups = frame["user_id"].astype(str).values

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss1.split(frame, groups=groups))
    trainval = frame.iloc[trainval_idx].copy()
    test = frame.iloc[test_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.17647058823529413, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss2.split(trainval, groups=trainval["user_id"].astype(str).values))
    train = trainval.iloc[train_idx].copy()
    val = trainval.iloc[val_idx].copy()

    return train, val, test

train_df, val_df, test_df = split_by_user(df)

print(f"      Train: {train_df.shape[0]} rows ({train_df['user_id'].nunique()} users)")
print(f"      Val:   {val_df.shape[0]} rows ({val_df['user_id'].nunique()} users)")
print(f"      Test:  {test_df.shape[0]} rows ({test_df['user_id'].nunique()} users)")

# ═══════════════════════════════════════════
# 3. SCALE FEATURES
# ═══════════════════════════════════════════

print("\n[3/7] Scaling features (RobustScaler)")

scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
scaler.fit(train_df[FEATURE_COLS].astype(float))

def apply_scaler(frame):
    out = frame.copy()
    out.loc[:, FEATURE_COLS] = scaler.transform(out[FEATURE_COLS].astype(float))
    return out

# Keep unscaled copies for pseudo-label computation
train_raw = train_df.copy()
val_raw = val_df.copy()
test_raw = test_df.copy()

train_df = apply_scaler(train_df)
val_df = apply_scaler(val_df)
test_df = apply_scaler(test_df)

with open(OUT_SCALER, "wb") as f:
    pickle.dump(scaler, f)
print(f"      Scaler saved -> {OUT_SCALER}")

# ═══════════════════════════════════════════
# 4. PSEUDO-LABELS (on raw unscaled data)
# ═══════════════════════════════════════════

print("\n[4/7] Creating pseudo-labels for cognitive load")

def robust_z(s):
    med = s.median()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = (q3 - q1) if (q3 - q1) != 0 else 1.0
    return (s - med) / iqr

def make_pseudo_load_scores(frame):
    out = frame.copy()
    score = (
        0.30 * robust_z(out["avg_dwell"]) +
        0.20 * robust_z(out["avg_flight"]) +
        0.20 * robust_z(out["pause_count"]) +
        0.15 * robust_z(out["backspace_count"]) +
        0.10 * robust_z(out["std_dwell"]) -
        0.25 * robust_z(out["typing_speed"])
    )
    out["pseudo_load_score"] = score
    out["pseudo_load_score"] = (
        out.groupby(["user_id", "session_id"], sort=False)["pseudo_load_score"]
           .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )
    return out

train_raw = make_pseudo_load_scores(train_raw)
val_raw = make_pseudo_load_scores(val_raw)
test_raw = make_pseudo_load_scores(test_raw)

q1 = train_raw["pseudo_load_score"].quantile(0.33)
q2 = train_raw["pseudo_load_score"].quantile(0.66)

def score_to_label(score):
    return np.select([score <= q1, score <= q2], [0, 1], default=2).astype(np.int32)

train_df["pseudo_label"] = score_to_label(train_raw["pseudo_load_score"])
val_df["pseudo_label"] = score_to_label(val_raw["pseudo_load_score"])
test_df["pseudo_label"] = score_to_label(test_raw["pseudo_load_score"])

for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    counts = d["pseudo_label"].value_counts().sort_index()
    print(f"      {name}: Low={counts.get(0,0)}, Med={counts.get(1,0)}, High={counts.get(2,0)}")

# ═══════════════════════════════════════════
# 5. BUILD SEQUENCES
# ═══════════════════════════════════════════

print("\n[5/7] Building sequences (window size = 4)")

def build_sequences_with_labels(frame, seq_len=SEQ_LEN):
    frame = frame.sort_values(
        ["user_id", "session_id", "window_id"], kind="mergesort"
    ).reset_index(drop=True)

    X_list, y_list = [], []

    for (_, _), g in frame.groupby(["user_id", "session_id"], sort=False):
        x = g[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = g["pseudo_label"].to_numpy(dtype=np.int32)

        if len(g) < seq_len:
            continue

        for i in range(len(g) - seq_len + 1):
            x_seq = x[i:i + seq_len]
            y_seq = y[i:i + seq_len]
            seq_label = np.bincount(y_seq, minlength=3).argmax()
            X_list.append(x_seq)
            y_list.append(seq_label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

X_train, y_train = build_sequences_with_labels(train_df)
X_val, y_val = build_sequences_with_labels(val_df)
X_test, y_test = build_sequences_with_labels(test_df)

print(f"      X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"      X_val:   {X_val.shape}  y_val:   {y_val.shape}")
print(f"      X_test:  {X_test.shape}  y_test:  {y_test.shape}")

# ═══════════════════════════════════════════
# 6. BUILD & TRAIN MODEL
# ═══════════════════════════════════════════

print("\n[6/7] Training GRU model")

# Class weights for imbalanced labels
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights_array)}
print(f"      Class weights: {class_weight_dict}")

# Build model
inputs = Input(shape=(SEQ_LEN, len(FEATURE_COLS)), name="input_seq")
x = GRU(64, return_sequences=True, name="gru_encoder")(inputs)
x = GlobalAveragePooling1D()(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(3, activation="softmax", name="load_class")(x)
model = Model(inputs, outputs)

early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)

# Phase 1: full training
print("\n      Phase 1: Training all layers (lr=1e-3, up to 10 epochs)")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Phase 2: fine-tune with lower learning rate
print("\n      Phase 2: Fine-tuning (lr=1e-4, up to 15 epochs)")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ═══════════════════════════════════════════
# 7. EVALUATE & SAVE
# ═══════════════════════════════════════════

print("\n[7/7] Evaluation on test set")

y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"\n      Accuracy: {round(acc * 100, 2)}%\n")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

cm = confusion_matrix(y_test, y_pred)
print("      Confusion Matrix:")
print(f"      {cm}\n")

# Save model
model.save(OUT_MODEL)
print(f"      Model saved  -> {OUT_MODEL}")
print(f"      Scaler saved -> {OUT_SCALER}")

# Optional: save accuracy plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc_hist = history_1.history["accuracy"] + history_2.history["accuracy"]
    val_acc_hist = history_1.history["val_accuracy"] + history_2.history["val_accuracy"]
    phase1_len = len(history_1.history["accuracy"])

    plt.figure(figsize=(8, 4))
    plt.plot(acc_hist, label="train_acc")
    plt.plot(val_acc_hist, label="val_acc")
    plt.axvline(x=phase1_len - 0.5, color="gray", linestyle="--", alpha=0.6, label="phase 2 start")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.savefig(os.path.join(BASE_DIR, "accuracy_plot.png"), dpi=150)
    print(f"      Plot saved   -> accuracy_plot.png")
except ImportError:
    print("      (matplotlib not installed, skipping plot)")

print("\n" + "=" * 50)
print("DONE! You can now run the app:")
print("      python app.py")
print("=" * 50)
