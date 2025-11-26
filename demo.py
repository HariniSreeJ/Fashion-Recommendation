

import os
import pickle
import numpy as np
import faiss
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# --------------------------- PATHS ---------------------------
SAVE_PATH     = "C:/ML_PROJ/models"
DF_PATH       = os.path.join(SAVE_PATH, "dataframe.pkl")
EMB_PATH      = os.path.join(SAVE_PATH, "embeddings.npy")
IDX_PATH      = os.path.join(SAVE_PATH, "faiss_index_cosine.bin")
TEST_IMG_PATH = "C:/ML_PROJ/sample_images/test_image.png"

# ------------------- CREATE DUMMY IMAGE IF MISSING -------------------
if not os.path.exists(TEST_IMG_PATH):
    print(f"Warning: Image not found: {TEST_IMG_PATH}")
    print("Creating dummy 224x224 black image...")
    os.makedirs(os.path.dirname(TEST_IMG_PATH), exist_ok=True)
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.imwrite(TEST_IMG_PATH, dummy)
    print(f"Dummy image saved to: {TEST_IMG_PATH}")

# ------------------- LOAD COMPONENTS -------------------
print("\n[INFO] Loading saved components...")

# 1. DataFrame
with open(DF_PATH, "rb") as f:
    df = pickle.load(f)
print(f"   → DataFrame: {len(df)} products")

# 2. Models
color_model = load_model(os.path.join(SAVE_PATH, "color_model.h5"))
type_model  = load_model(os.path.join(SAVE_PATH, "type_model.h5"))
print("   → Models loaded")

# 3. Encoders
with open(os.path.join(SAVE_PATH, "color_encoder.pkl"), "rb") as f:
    color_encoder = pickle.load(f)
with open(os.path.join(SAVE_PATH, "type_encoder.pkl"), "rb") as f:
    type_encoder = pickle.load(f)
print("   → Encoders loaded")

# 4. Embeddings
embeddings = np.load(EMB_PATH).astype(np.float32)
embeddings = np.ascontiguousarray(embeddings)
print(f"   → Embeddings: {embeddings.shape}")

# ------------------- FAISS INDEX (in-memory) -------------------
if os.path.exists(IDX_PATH):
    print(f"   → Loading FAISS index from: {IDX_PATH}")
    index = faiss.read_index(IDX_PATH)
else:
    print("   → Building FAISS index in memory (cosine)...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_norm = embeddings / norms
    embeddings_norm = np.ascontiguousarray(embeddings_norm)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_norm)
    print(f"   → In-memory index built: {index.ntotal} vectors")

print(f"   → FAISS ready: {index.ntotal} vectors")

# ------------------- IMAGE PREPROCESS -------------------
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

print(f"\n[INFO] Loading test image: {TEST_IMG_PATH}")
test_img = preprocess_image(TEST_IMG_PATH)

# ------------------- PREDICT COLOUR & TYPE -------------------
c_pred = color_model.predict(test_img, verbose=0)[0]
t_pred = type_model.predict(test_img, verbose=0)[0]

pred_color = color_encoder.classes_[np.argmax(c_pred)]
pred_type  = type_encoder.classes_[np.argmax(t_pred)]

conf_color = np.max(c_pred)
conf_type  = np.max(t_pred)

print(f"Predicted → Colour: {pred_color} ({conf_color:.1%}) | Type: {pred_type} ({conf_type:.1%})")

# ------------------- RECOMMEND -------------------
def recommend_from_prediction(color, article_type, top_k=5):
    cand = df[(df['baseColour'] == color) & (df['articleType'] == article_type)]
    if len(cand) == 0:
        print("No products found for this colour + type.")
        return None

    cand_idx = cand.index.values
    mean_vec = embeddings[cand_idx].mean(axis=0).reshape(1, -1).astype(np.float32)
    mean_vec = np.ascontiguousarray(mean_vec)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec /= norm

    _, I = index.search(mean_vec, top_k * 5)
    rec_idx = [i for i in I[0] if i in cand_idx][:top_k]

    if len(rec_idx) == 0:
        return None

    result = df.iloc[rec_idx][['id', 'productDisplayName', 'baseColour', 'articleType']]
    return result.reset_index(drop=True)

# ------------------- RUN RECOMMENDATION -------------------
print("\n" + "="*70)
print("RECOMMENDATIONS FOR YOUR IMAGE")
print("="*70)

rec = recommend_from_prediction(pred_color, pred_type, top_k=5)

if rec is not None and len(rec) > 0:
    print("\nRecommended Items:")
    print(rec.to_string(index=False))
else:
    print("No similar items found.")