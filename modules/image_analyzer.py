# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os
# import json
# import cv2
# import numpy as np
# from PIL import Image
# import io

# # --- CONFIGURATION ---
# DATASET_PATH = 'plantvillage dataset/'  # Folder containing your curated PlantVillage folders
# MODEL_SAVE_NAME = 'agrilink_quality_v1.h5'
# LABEL_MAP_NAME = 'class_indices.json'
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32
# EPOCHS = 10

# def analyze_image(image_bytes: bytes, crop_type: str) -> dict:
#     """
#     Analyze crop image quality using OpenCV HSV analysis.
#     Returns quality grade, freshness score, and recommendations.
#     """
#     # Convert bytes to numpy array
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     if img is None:
#         return {
#             "quality_grade": "C",
#             "grade_label": "Unable to analyze",
#             "freshness_score": 0,
#             "defect_score": 0,
#             "overall_score": 0,
#             "shelf_life_days": 0,
#             "recommendation": "Image could not be processed"
#         }
    
#     # Convert to HSV for color analysis
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
#     # Calculate saturation and value metrics
#     saturation = np.mean(hsv[:, :, 1])
#     value = np.mean(hsv[:, :, 2])
    
#     # Simple freshness heuristic based on color vibrancy
#     freshness_score = min(100, (saturation / 255) * 150 + (value / 255) * 50)
    
#     # Defect detection (simplified - looks for dark spots)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     dark_pixels = np.sum(gray < 50) / gray.size
#     defect_score = max(0, 100 - (dark_pixels * 500))
    
#     # Overall score
#     overall_score = (freshness_score * 0.6 + defect_score * 0.4)
    
#     # Determine grade
#     if overall_score >= 75:
#         grade = "A"
#         label = "Premium Quality"
#         shelf_life = 7
#     elif overall_score >= 50:
#         grade = "B"
#         label = "Standard Quality"
#         shelf_life = 4
#     else:
#         grade = "C"
#         label = "Below Standard"
#         shelf_life = 2
    
#     # Recommendation
#     if grade == "A":
#         recommendation = "Excellent quality. Suitable for premium markets and export."
#     elif grade == "B":
#         recommendation = "Good quality. Suitable for local markets and retail."
#     else:
#         recommendation = "Quality concerns detected. Consider processing or immediate sale."
    
#     return {
#         "quality_grade": grade,
#         "grade_label": label,
#         "freshness_score": round(freshness_score, 1),
#         "defect_score": round(defect_score, 1),
#         "overall_score": round(overall_score, 1),
#         "shelf_life_days": shelf_life,
#         "recommendation": recommendation
#     }

# def train_model():
#     # 1. PREPROCESSING & DATA AUGMENTATION
#     print("Pre-processing images...")
#     datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         horizontal_flip=True,
#         validation_split=0.2
#     )

#     train_data = datagen.flow_from_directory(
#         DATASET_PATH,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='training'
#     )

#     val_data = datagen.flow_from_directory(
#         DATASET_PATH,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='validation'
#     )

#     # Save class indices to a JSON file so the App knows which index is which crop
#     with open(LABEL_MAP_NAME, 'w') as f:
#         json.dump(train_data.class_indices, f)
#     print(f"Labels saved to {LABEL_MAP_NAME}")

#     # 2. DEFINE MODEL (Transfer Learning)
#     print("Building model with MobileNetV2 base...")
#     base_model = tf.keras.applications.MobileNetV2(
#         input_shape=(224, 224, 3), 
#         include_top=False, 
#         weights='imagenet'
#     )
#     base_model.trainable = False  # Freeze pre-trained weights

#     model = models.Sequential([
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(128, activation='relu'),
#         layers.Dropout(0.3),
#         layers.Dense(train_data.num_classes, activation='softmax')
#     ])

#     model.compile(
#         optimizer='adam', 
#         loss='categorical_crossentropy', 
#         metrics=['accuracy']
#     )

#     # 3. START TRAINING
#     print("Starting training loop...")
#     history = model.fit(
#         train_data,
#         validation_data=val_data,
#         epochs=EPOCHS
#     )

#     # 4. SAVE MODEL
#     model.save(MODEL_SAVE_NAME)
#     print(f"Model saved successfully as {MODEL_SAVE_NAME}")

# if __name__ == "__main__":
#     # Check if data directory exists before starting
#     if os.path.exists(DATASET_PATH):
#         train_model()
#     else:
#         print(f"Error: Dataset folder '{DATASET_PATH}' not found. Please create it and add crop folders.")