#  Visual Fashion Recommendation System

##  Overview
A deep learning–based visual fashion recommendation system that analyzes clothing images, extracts key attributes, and retrieves visually similar products from a large catalog.  
The system supports both **image-based** and **text-based** queries for an enhanced fashion discovery experience.

---

## Abstract
This project uses deep learning to understand fashion product images and recommend similar items.  
It automatically identifies clothing attributes such as **color** and **category**, and retrieves visually similar products using a fast similarity search engine.  
The system enables **real-time**, intelligent recommendations suitable for e-commerce and fashion platforms.

---

##  Core Components

### 🔹 Attribute Recognition
- Twin **ResNet50** models to predict garment **color** and **category**.

### 🔹 Visual Search Engine
- **FAISS** index for lightning-fast similarity-based retrieval.

### 🔹 Dual-Mode Queries
- Accepts **user-uploaded images**.  
- Accepts **text descriptions** mapped to visual embeddings.

---

##  Key Innovations
- **Automated Tagging**: Eliminates the need for manual product labeling using CNN models.  
- **Hybrid Filtering**: Combines attribute matching and visual similarity for high-quality recommendations.  
- **Deployment-Ready**: Complete end-to-end pipeline designed for real-time inference.

---

##  Applications
- “**Shop-the-Look**” features for e-commerce  
- **Visually similar** product suggestions  
- Fashion **catalog management**  
- **Personalized styling** and recommendation apps  

---

##  Technologies Used
- **Python**  
- **TensorFlow**  
- **OpenCV**  
- **FAISS**  
- **Scikit-learn**  

---

