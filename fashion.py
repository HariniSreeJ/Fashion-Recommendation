import os

# Base directory where you unzipped the dataset
data_dir = "C:/ML_PROJ/myntradataset"

# Path to the images folder
images_dir = os.path.join(data_dir, "images")

# Path to the CSV file
csv_path = os.path.join(data_dir, "styles.csv")

# Check if paths exist
print("Images directory exists:", os.path.exists(images_dir))
print("CSV file exists:", os.path.exists(csv_path))

# ==============================================================
# 2.  LOAD & PRE-PROCESS DATA
# ==============================================================

#import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

SAVE_PATH = "C:/ML_PROJ/models/"
os.makedirs(SAVE_PATH, exist_ok=True)

# ---- read CSV -------------------------------------------------
df = pd.read_csv(csv_path, on_bad_lines='skip')
df['image_path'] = df['id'].astype(str) + '.jpg'
df['full_path']  = df['image_path'].apply(lambda x: os.path.join(images_dir, x))

# ---- keep only images that really exist ----------------------
df = df[df['full_path'].apply(os.path.exists)].reset_index(drop=True)

# ---- SAMPLE (speed up training – remove for full dataset) ----
SAMPLE_SIZE = 5000                     # change to None for full
if SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)

print(f"Working with {len(df)} products")

# ---- label encoders -------------------------------------------
color_encoder = LabelEncoder()
df['color_label'] = color_encoder.fit_transform(df['baseColour'].fillna('unknown'))

type_encoder = LabelEncoder()
df['type_label'] = type_encoder.fit_transform(df['articleType'].fillna('unknown'))

num_colors = len(color_encoder.classes_)
num_types  = len(type_encoder.classes_)
print(f"Colors: {num_colors} | Types: {num_types}")

# --------------------------------------------------------------
#  FIX: Filter out rare classes BEFORE creating arrays
# --------------------------------------------------------------
counts = df['color_label'].value_counts()
df = df[df['color_label'].isin(counts[counts > 1].index)].reset_index(drop=True)
print(f"After filtering rare colors: {len(df)} samples remain")

# --------------------------------------------------------------
# Image preprocessing function
# --------------------------------------------------------------
def preprocess_image(path):
    try:
        img = load_img(path, target_size=(224, 224))
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        return preprocess_input(arr)[0]
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return np.zeros((224, 224, 3), dtype=np.float32)

# ---- Build feature and label arrays ---------------------------
X = np.stack([preprocess_image(p) for p in df['full_path']])

y_color = pd.get_dummies(df['color_label']).values
y_type  = pd.get_dummies(df['type_label']).values

# Quick sanity check
print(f"Shapes -> X: {X.shape}, y_color: {y_color.shape}, y_type: {y_type.shape}")

# --------------------------------------------------------------
# TRAIN / TEST SPLIT – SAFE VERSION
# --------------------------------------------------------------
X_train, X_test, y_color_train, y_color_test, y_type_train, y_type_test = train_test_split(
    X, y_color, y_type,
    test_size=0.2,
    random_state=42,
    stratify=df['color_label']
)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

np.save(os.path.join(SAVE_PATH, "X_all.npy"), X)
np.save(os.path.join(SAVE_PATH, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_PATH, "y_color_test.npy"), y_color_test)
np.save(os.path.join(SAVE_PATH, "y_type_test.npy"), y_type_test)

# Save label encoders
with open(os.path.join(SAVE_PATH, "color_encoder.pkl"), "wb") as f:
    pickle.dump(color_encoder, f)
with open(os.path.join(SAVE_PATH, "type_encoder.pkl"), "wb") as f:
    pickle.dump(type_encoder, f)



# ==============================================================
# 3.  BUILD RESNET50 MODELS (SHADE + ITEM TYPE)
# ==============================================================

#  RECOMPUTE AFTER FILTERING
num_colors = df['color_label'].nunique()
num_types  = df['type_label'].nunique()
print(f"Updated Colors: {num_colors} | Updated Types: {num_types}")

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

def build_resnet(num_classes, name):
    base = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(224, 224, 3))
    base.trainable = False                     # freeze pretrained weights

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ], name=name)

    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---- colour model --------------------------------------------
color_model = build_resnet(num_colors, "ColorModel")
color_model.summary()

# ---- type model ----------------------------------------------
type_model = build_resnet(num_types, "TypeModel")
type_model.summary()


# ==============================================================
# 4.  TRAIN COLOUR MODEL (epochs + plots)
# ==============================================================

import matplotlib.pyplot as plt

EPOCHS = 20
BATCH  = 32

# ---- Train the colour model ----------------------------------
hist_color = color_model.fit(
    X_train, y_color_train,
    validation_data=(X_test, y_color_test),
    epochs=EPOCHS,
    batch_size=BATCH,
    verbose=2
)

# ---- Plot training progress ----------------------------------
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(hist_color.history['accuracy'], label='Train Acc')
plt.plot(hist_color.history['val_accuracy'], label='Val Acc')
plt.title('Colour Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(hist_color.history['loss'], label='Train Loss')
plt.plot(hist_color.history['val_loss'], label='Val Loss')
plt.title('Colour Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
# ---- save accuracy/loss plots ----
plt.savefig("C:/ML_PROJ/outputs/color_training_plot.png")


print(f"Final Validation Accuracy (Colour): {hist_color.history['val_accuracy'][-1]:.4f}")
# ==============================================================
# 5. SAVE TRAINED MODELS
# ==============================================================

import os

# make sure model and output folders exist
os.makedirs("C:/ML_PROJ/models", exist_ok=True)
os.makedirs("C:/ML_PROJ/outputs", exist_ok=True)

# ---- save trained models ----
color_model.save("C:/ML_PROJ/models/color_model.h5")
type_model.save("C:/ML_PROJ/models/type_model.h5")

print("Models saved to: C:/ML_PROJ/models/")

# ---- save training history (optional) ----
import json
with open("C:/ML_PROJ/outputs/color_training_history.json", "w") as f:
    json.dump(hist_color.history, f)
    
    
    
# ==============================================================
# 5. TRAIN TYPE MODEL (Initial + Fine-tuning)
# ==============================================================

from tensorflow.keras.callbacks import ModelCheckpoint
import os, pickle, matplotlib.pyplot as plt

os.makedirs("C:/ML_PROJ/models", exist_ok=True)

# --- Save checkpoints ---
type_ckpt_path = "C:/ML_PROJ/models/type_model_best.h5"
color_ckpt_path = "C:/ML_PROJ/models/color_model_best.h5"

checkpoint_type = ModelCheckpoint(
    filepath=type_ckpt_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

print("\n[INFO] Training TYPE model (initial phase)...\n")

hist_type = type_model.fit(
    X_train, y_type_train,
    validation_data=(X_test, y_type_test),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=[checkpoint_type],
    verbose=2
)

# ---- Save final model & history ----
type_model.save("C:/ML_PROJ/models/type_model_initial.h5")
with open("C:/ML_PROJ/models/type_history_initial.pkl", "wb") as f:
    pickle.dump(hist_type.history, f)

# ---- Plot accuracy/loss ----
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(hist_type.history['accuracy'], label='Train Acc')
plt.plot(hist_type.history['val_accuracy'], label='Val Acc')
plt.title('Type Model Accuracy (Initial)')
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist_type.history['loss'], label='Train Loss')
plt.plot(hist_type.history['val_loss'], label='Val Loss')
plt.title('Type Model Loss (Initial)')
plt.legend()
plt.savefig("C:/ML_PROJ/models/type_model_initial_plot.png")
plt.close()

print(f"Initial Val Accuracy (Type): {hist_type.history['val_accuracy'][-1]:.4f}")

# ==============================================================
# 5b. FINE-TUNE LAST 30 LAYERS OF BOTH MODELS
# ==============================================================

from tensorflow.keras import optimizers

def fine_tune_model(model, lr=1e-5, unfreeze_layers=30):
    print(f"\n[INFO] Fine-tuning last {unfreeze_layers} layers...\n")
    for layer in model.layers[0].layers[-unfreeze_layers:]:
        layer.trainable = True
    model.compile(optimizer=optimizers.Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Fine-tune both models
color_model = fine_tune_model(color_model)
type_model  = fine_tune_model(type_model)

print("\n[INFO] Fine-tuning both models for 5 more epochs...\n")

EPOCHS_FINE = 5

hist_color_ft = color_model.fit(
    X_train, y_color_train,
    validation_data=(X_test, y_color_test),
    epochs=EPOCHS_FINE, batch_size=BATCH, verbose=2)

hist_type_ft = type_model.fit(
    X_train, y_type_train,
    validation_data=(X_test, y_type_test),
    epochs=EPOCHS_FINE, batch_size=BATCH, verbose=2)

# ---- Save fine-tuned models ----
color_model.save("C:/ML_PROJ/models/color_model_finetuned.h5")
type_model.save("C:/ML_PROJ/models/type_model_finetuned.h5")

with open("C:/ML_PROJ/models/type_history_finetune.pkl", "wb") as f:
    pickle.dump(hist_type_ft.history, f)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(hist_type_ft.history['accuracy'], label='Train Acc')
plt.plot(hist_type_ft.history['val_accuracy'], label='Val Acc')
plt.title('Type Model Accuracy (Fine-tuned)')
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist_type_ft.history['loss'], label='Train Loss')
plt.plot(hist_type_ft.history['val_loss'], label='Val Loss')
plt.title('Type Model Loss (Fine-tuned)')
plt.legend()
plt.savefig("C:/ML_PROJ/models/type_model_finetuned_plot.png")
plt.close()

print(f"Final Val Accuracy (Fine-tuned TYPE): {hist_type_ft.history['val_accuracy'][-1]:.4f}")
# ==============================================================
# 6. CONFUSION MATRIX (Colour) + EVALUATION
# ==============================================================
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# ------------------- Load saved components -------------------
SAVE_PATH = "C:/ML_PROJ/models/"
os.makedirs(SAVE_PATH, exist_ok=True)

# Load models
color_model = load_model(os.path.join(SAVE_PATH, "color_model.h5"))
type_model  = load_model(os.path.join(SAVE_PATH, "type_model.h5"))

# Load encoders
with open(os.path.join(SAVE_PATH, "color_encoder.pkl"), "rb") as f:
    color_encoder = pickle.load(f)

with open(os.path.join(SAVE_PATH, "type_encoder.pkl"), "rb") as f:
    type_encoder = pickle.load(f)

# Load data arrays
X_test        = np.load(os.path.join(SAVE_PATH, "X_test.npy"))
y_color_test  = np.load(os.path.join(SAVE_PATH, "y_color_test.npy"))
y_type_test   = np.load(os.path.join(SAVE_PATH, "y_type_test.npy"))
X              = np.load(os.path.join(SAVE_PATH, "X_all.npy"))

# Load dataframe
df = pickle.load(open(os.path.join(SAVE_PATH, "dataframe.pkl"), "rb"))

# ------------------- Evaluate Colour Model -------------------
print("\n[INFO] Evaluating COLOUR model...\n")

y_pred = color_model.predict(X_test, verbose=0)
y_pred_cls = np.argmax(y_pred, axis=1)
y_true_cls = np.argmax(y_color_test, axis=1)

# ensure alignment of labels and encoder classes
labels = np.unique(np.concatenate([y_true_cls, y_pred_cls]))

report = classification_report(
    y_true_cls,
    y_pred_cls,
    labels=labels,
    target_names=color_encoder.classes_[labels],
    digits=4,
    zero_division=0
)
print(report)

# ---- Confusion Matrix (Top-12) ----
top_k = 12
cm = confusion_matrix(y_true_cls, y_pred_cls)

counts = np.bincount(y_true_cls)
sorted_idx = np.argsort(-counts)
valid_classes = np.arange(cm.shape[0])
idx = [i for i in sorted_idx if i in valid_classes][:top_k]

cm_top = cm[idx, :][:, idx]

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm_top,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=color_encoder.classes_[idx],
    yticklabels=color_encoder.classes_[idx]
)
plt.title('Confusion Matrix – Top 12 Colours')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "color_confusion_matrix.png"))
plt.close()
print(" Confusion matrix saved successfully.")

# ==============================================================
# 7. EXTRACT EMBEDDINGS
# ==============================================================
from tensorflow.keras import models

print("\n[INFO] Extracting embeddings for FAISS index...\n")

embedding_layer = color_model.layers[-4].output
embed_model = models.Model(inputs=color_model.input, outputs=embedding_layer)

embeddings = embed_model.predict(X, batch_size=32, verbose=1)
np.save(os.path.join(SAVE_PATH, "embeddings.npy"), embeddings)
print(" Embeddings extracted and saved. Shape:", embeddings.shape)

# ==============================================================
# 8. BUILD FAISS INDEX
# ==============================================================
import faiss

print("\n[INFO] Building FAISS index...\n")

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings.astype(np.float32))
faiss.write_index(index, os.path.join(SAVE_PATH, "faiss_index.bin"))
print(f" FAISS index ready – {index.ntotal} vectors")

# ==============================================================
# 9. NLP PARSER
# ==============================================================
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("\n[INFO] Loading NLP model...\n")

nlp = SentenceTransformer('all-MiniLM-L6-v2')
color_texts = list(color_encoder.classes_)
type_texts  = list(type_encoder.classes_)
color_embs = nlp.encode(color_texts, show_progress_bar=False)
type_embs  = nlp.encode(type_texts, show_progress_bar=False)

def parse_query(query: str):
    q_emb = nlp.encode([query])
    sim_c = cosine_similarity(q_emb, color_embs)[0]
    sim_t = cosine_similarity(q_emb, type_embs)[0]
    best_c = color_encoder.classes_[np.argmax(sim_c)]
    best_t = type_encoder.classes_[np.argmax(sim_t)]
    return best_c, best_t

# ==============================================================
# 10. RECOMMENDATION FUNCTION
# ==============================================================
def recommend(query: str, top_k: int = 5):
    col, typ = parse_query(query)
    print(f"Parsed → Colour: {col} | Type: {typ}")

    cand = df[(df['baseColour'] == col) & (df['articleType'] == typ)]
    if len(cand) == 0:
        return "No exact colour+type match."

    cand_idx = cand.index.values
    mean_vec = embeddings[cand_idx].mean(axis=0).reshape(1, -1).astype(np.float32)
    _, I = index.search(mean_vec, top_k * 5)
    rec_idx = [i for i in I[0] if i in cand_idx][:top_k]
    rec = df.iloc[rec_idx][['id', 'productDisplayName', 'baseColour', 'articleType']]
    return rec.reset_index(drop=True)

# ------------------- Sample Query Test -------------------
query = "navy blue tshirt"
print("\n[INFO] Running sample recommendation...\n")
print(recommend(query))

# ==============================================================
# 11. INFERENCE ON NEW IMAGE
# ==============================================================
import cv2

def preprocess_image_local(path, size=(224, 224)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype('float32') / 255.0
    return img

new_path = "C:/ML_PROJ/sample_images/test_img.jpg"  # change path
img = preprocess_image_local(new_path)
img_batch = np.expand_dims(img, axis=0)

c_pred = color_model.predict(img_batch, verbose=0)
c_label = color_encoder.classes_[np.argmax(c_pred)]

t_pred = type_model.predict(img_batch, verbose=0)
t_label = type_encoder.classes_[np.argmax(t_pred)]

print(f"\nDetected Shade: {c_label} | Type: {t_label}")
rec_query = f"{c_label} {t_label}"
print("\nRecommendations for this image:")
print(recommend(rec_query))



import os
import pickle
import numpy as np
import faiss
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SAVE_PATH = "C:/ML_PROJ/models/"

print("[INFO] Loading components...")
with open(os.path.join(SAVE_PATH, "dataframe.pkl"), "rb") as f:
    df = pickle.load(f)

color_model = load_model(os.path.join(SAVE_PATH, "color_model.h5"))
type_model = load_model(os.path.join(SAVE_PATH, "type_model.h5"))

with open(os.path.join(SAVE_PATH, "color_encoder.pkl"), "rb") as f:
    color_encoder = pickle.load(f)
with open(os.path.join(SAVE_PATH, "type_encoder.pkl"), "rb") as f:
    type_encoder = pickle.load(f)

embeddings = np.load(os.path.join(SAVE_PATH, "embeddings.npy")).astype(np.float32)
embeddings = np.ascontiguousarray(embeddings)

# Manual L2 normalization
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1.0
embeddings_norm = embeddings / norms
embeddings_norm = np.ascontiguousarray(embeddings_norm)

# FAISS Index (Cosine)
faiss_path = os.path.join(SAVE_PATH, "faiss_index_cosine.bin")
dim = embeddings.shape[1]

if os.path.exists(faiss_path):
    index = faiss.read_index(faiss_path)
    print(f"FA Faith index loaded: {index.ntotal} vectors")
else:
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_norm)
    faiss.write_index(index, faiss_path)
    print("Index saved")

# NLP
print("[INFO] Loading SentenceTransformer...")
nlp = SentenceTransformer('all-MiniLM-L6-v2')
color_embs = nlp.encode(list(color_encoder.classes_), convert_to_numpy=True)
type_embs = nlp.encode(list(type_encoder.classes_), convert_to_numpy=True)

def parse_query(q):
    qe = nlp.encode([q], convert_to_numpy=True)
    c = color_encoder.classes_[np.argmax(cosine_similarity(qe, color_embs))]
    t = type_encoder.classes_[np.argmax(cosine_similarity(qe, type_embs))]
    return c, t

def recommend(q, k=5):
    c, t = parse_query(q)
    print(f"→ {c} {t}")
    cand = df[(df['baseColour'] == c) & (df['articleType'] == t)]
    if len(cand) == 0: return None
    idx = cand.index.values
    vec = embeddings[idx].mean(0).reshape(1, -1).astype(np.float32)
    vec /= np.linalg.norm(vec) or 1
    vec = np.ascontiguousarray(vec)
    _, I = index.search(vec, k*5)
    rec = [i for i in I[0] if i in idx][:k]
    return df.iloc[rec][['id', 'productDisplayName', 'baseColour', 'articleType']].reset_index(drop=True)

# TEST
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST: navy blue tshirt")
    print("="*60)
    res = recommend("navy blue tshirt")
    if res is not None:
        print("\nRecommended:")
        print(res.to_string(index=False))
        
"""      
# fashion_demo.py
# --------------------------------------------------------------
#  Simple demo – no sentence-transformers, no torch, no errors
# --------------------------------------------------------------

import os
import pickle
import numpy as np
import faiss

# --------------------------------------------------------------
# 1. Paths
# --------------------------------------------------------------
SAVE_PATH = "C:/ML_PROJ/models/"
DF_PATH   = os.path.join(SAVE_PATH, "dataframe.pkl")
EMB_PATH  = os.path.join(SAVE_PATH, "embeddings.npy")
IDX_PATH  = os.path.join(SAVE_PATH, "faiss_index_cosine.bin")

# --------------------------------------------------------------
# 2. Load data
# --------------------------------------------------------------
print("[INFO] Loading dataframe …")
with open(DF_PATH, "rb") as f:
    df = pickle.load(f)

print("[INFO] Loading embeddings …")
embeddings = np.load(EMB_PATH).astype(np.float32)
embeddings = np.ascontiguousarray(embeddings)          # safety

print("[INFO] Loading FAISS index …")
index = faiss.read_index(IDX_PATH)
print(f"FAISS index ready – {index.ntotal} vectors")

# --------------------------------------------------------------
# 3. Very small query → (colour, type) mapper
# --------------------------------------------------------------
QUERY_MAP = {
    "navy blue tshirt":   ("Navy",      "Tshirts"),
    "red jacket":         ("Red",       "Jackets"),
    "black formal shoes": ("Black",     "Casual Shoes"),
    # add more entries here if you like
}

# --------------------------------------------------------------
# 4. Recommendation function
# --------------------------------------------------------------
def recommend(query: str, top_k: int = 5):
    # ----- resolve colour / type -----
    if query.lower() not in QUERY_MAP:
        print("Warning: Unknown query – falling back to first entry")
        col, typ = next(iter(QUERY_MAP.values()))
    else:
        col, typ = QUERY_MAP[query.lower()]

    print(f"Parsed → Colour: {col} | Type: {typ}")

    # ----- candidates -----
    cand = df[(df['baseColour'] == col) & (df['articleType'] == typ)]
    if len(cand) == 0:
        print("No products match colour+type")
        return None

    cand_idx = cand.index.values

    # ----- mean vector (already L2-normalised in the index) -----
    mean_vec = embeddings[cand_idx].mean(axis=0).reshape(1, -1).astype(np.float32)
    mean_vec = np.ascontiguousarray(mean_vec)
    # normalise query vector (cosine = inner product of unit vectors)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec /= norm

    # ----- FAISS search -----
    _, I = index.search(mean_vec, top_k * 5)

    # ----- keep only candidates -----
    rec_idx = [i for i in I[0] if i in cand_idx][:top_k]

    if not rec_idx:
        print("No similar items inside the candidate set")
        return None

    result = df.iloc[rec_idx][['id', 'productDisplayName', 'baseColour', 'articleType']]
    return result.reset_index(drop=True)

# --------------------------------------------------------------
# 5. Run demo
# --------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DEMO: navy blue tshirt")
    print("="*60)

    res = recommend("navy blue tshirt", top_k=5)

    if res is not None and len(res) > 0:
        print("\nRecommended items:")
        print(res.to_string(index=False))
    else:
        print("No recommendations")
        """