# cross_modal_retrieval_final.py
import streamlit as st
import torch
import clip
from PIL import Image
import os
import numpy as np

# -------------------------
# Load CLIP Model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# -------------------------
# Encode all images in folder
# -------------------------
@st.cache_resource
def load_image_features(folder):
    image_paths = []
    image_features = []

    if not os.path.exists(folder):
        return [], None

    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, fname)
            try:
                image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    feature = model.encode_image(image)
                    feature /= feature.norm(dim=-1, keepdim=True)
                image_paths.append(path)
                image_features.append(feature)
            except Exception as e:
                print(f"Skipping {fname}: {e}")

    if len(image_features) == 0:
        return [], None

    image_features = torch.cat(image_features, dim=0)
    return image_paths, image_features


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Smart Cross-Modal Retrieval", layout="centered")
st.title("🖼️ Smart Image Retrieval System using CLIP")

# Folder input
image_folder = st.text_input("Enter image folder path:", "images")

# Threshold controls
confidence_threshold = st.slider("Minimum confidence threshold", 0.0, 1.0, 0.3, 0.01)
similarity_gap = st.slider("Max similarity gap (difference from top score)", 0.0, 0.5, 0.1, 0.01)

query = st.text_input("Enter your text query (e.g., 'a cute furry pet playing in the park'):")

if not os.path.exists(image_folder):
    st.warning("⚠️ Folder not found! Please enter a valid image folder.")
    st.stop()

with st.spinner("🔍 Encoding images..."):
    image_paths, image_features = load_image_features(image_folder)

if image_features is None or len(image_paths) == 0:
    st.error("No valid images found in the folder.")
    st.stop()

if st.button("Search") and query.strip():
    # Encode text
    tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_feature = model.encode_text(tokens)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)

    # Compute cosine similarities
    similarities = (image_features @ text_feature.T).squeeze(1).cpu().numpy()

    # Normalize to [0, 1]
    if similarities.max() - similarities.min() > 1e-8:
        similarities_norm = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    else:
        similarities_norm = np.zeros_like(similarities)

    # Sort images by similarity
    sorted_indices = np.argsort(similarities_norm)[::-1]
    top_score = similarities_norm[sorted_indices[0]]

    # Filter: only include if within gap and above threshold
    selected_indices = [
        i for i in sorted_indices
        if (similarities_norm[i] >= confidence_threshold) and 
           (top_score - similarities_norm[i] <= similarity_gap)
    ]

    if len(selected_indices) == 0:
        st.warning(f"No relevant images found (Top score {top_score:.2f}). Try lowering thresholds.")
    else:
        st.success(f"✅ Found {len(selected_indices)} relevant image(s) (Top confidence: {top_score:.2f})")
        for idx in selected_indices:
            st.image(image_paths[idx], use_column_width=True,
                     caption=f"{os.path.basename(image_paths[idx])} — confidence {similarities_norm[idx]:.2f}")
