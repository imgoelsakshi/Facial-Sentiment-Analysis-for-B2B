# **MindMint – Real-Time Facial Emotion Recognition for Focus Groups**

MindMint is a lightweight deep-learning–based facial emotion recognition tool designed to help startups gather richer customer insights during focus groups.
By analyzing facial expressions in real time, moderators can better understand user reactions and enhance overall customer experience.

## ** Project Goal**

To support startups in the pre-final stages by providing **unbiased, real-time emotional insights** during customer feedback sessions—going beyond traditional surveys, interviews, and forms.

## ** Key Features**

* **Real-time facial emotion recognition** using a custom CNN model
* **Lightweight deep learning architecture** with fewer parameters
* **Higher training efficiency without compromising accuracy**
* **Single and multi-face detection** using Haar Cascade
* **Emotion classification**: angry, fearful, happy, neutral, sad, surprised
* **Frame-sequence–based analysis** for richer temporal understanding
* **Designed specifically for focus group moderators**

## ** Technical Overview**

### **Model Architecture**

* Custom CNN model optimized for lean structure
* Trained on a **modified FER2013 dataset** (20,000+ images, 48×48 pixels)
* Real-time classification pipeline using sequences of frames

### **Training Data**

* FER2013 base dataset
* Modified version (removed *disgust*)
* Participants aged 23–58
* Face-to-camera distance ~60 cm

### **Pipeline**

1. Capture frames
2. Detect faces using Haar Cascade
3. Crop & preprocess images
4. Apply image augmentation
5. Feed sequences into CNN
6. Generate real-time emotion predictions

---

## ** Model Performance**

| Metric                  | Accuracy   |
| ----------------------- | ---------- |
| **Training Accuracy**   | **95.85%** |
| **Validation Accuracy** | **89.64%** |

## ** Tech Stack**

* **Languages:** Python
* **Libraries:** Keras, NumPy, Pandas, Pygame
* **Tools/Methods:** Haar Cascade, ImageDataGenerator, CNN

## ** Challenges**

* Tuning model parameters for competitive accuracy
* Handling variations in color images and backgrounds
* Balancing model size with performance in a competitive domain

## ** Future Enhancements**

* Fully online version using smartphone front cameras
* Gamified user-feedback interface for larger datasets
* GAN-based model improvement system
* Integration of reinforcement learning for smarter data collection

## **📌 Use Case Summary**

MindMint helps startups who must collect customer feedback from focus groups by providing **deep emotional insights**, unlike standard surveys or interview-based methods.
It helps teams understand user reactions more accurately and improve feature design, UI, and overall customer experience.
