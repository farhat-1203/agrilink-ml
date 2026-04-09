import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

# --- CONFIGURATION ---
DATASET_PATH = 'plantvillage dataset/'  # Folder containing your curated PlantVillage folders
MODEL_SAVE_NAME = 'agrilink_quality_v1.h5'
LABEL_MAP_NAME = 'class_indices.json'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

def train_model():
    # 1. PREPROCESSING & DATA AUGMENTATION
    print("Pre-processing images...")
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Save class indices to a JSON file so the App knows which index is which crop
    with open(LABEL_MAP_NAME, 'w') as f:
        json.dump(train_data.class_indices, f)
    print(f"Labels saved to {LABEL_MAP_NAME}")

    # 2. DEFINE MODEL (Transfer Learning)
    print("Building model with MobileNetV2 base...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze pre-trained weights

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    # 3. START TRAINING
    print("Starting training loop...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    # 4. SAVE MODEL
    model.save(MODEL_SAVE_NAME)
    print(f"Model saved successfully as {MODEL_SAVE_NAME}")

if __name__ == "__main__":
    # Check if data directory exists before starting
    if os.path.exists(DATASET_PATH):
        train_model()
    else:
        print(f"Error: Dataset folder '{DATASET_PATH}' not found. Please create it and add crop folders.")