import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Rescaling, RandomFlip, RandomRotation
import os
import numpy as np

# --- Configuration ---
# Define paths based on your provided folder structure
base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'FER-2013')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Model and image parameters
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64
EPOCHS = 50 # Increase for better accuracy, but longer training time

# --- Data Loading and Preprocessing ---
# Use image_dataset_from_directory to load images from the folders
print("Loading training data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True
)

print("Loading validation data...")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=False
)

# Get class names from the directory structure (e.g., ['angry', 'happy', ...])
class_names = train_dataset.class_names
num_classes = len(class_names)
print("Found class names:", class_names)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# --- Model Definition ---
# Data augmentation layers to prevent overfitting
data_augmentation = Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.1),
])

# Build the CNN model
model = Sequential([
    # Input layer - rescale pixel values from [0, 255] to [0, 1]
    Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    
    # Data Augmentation
    data_augmentation,
    
    # Convolutional Block 1
    Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'),
    MaxPooling2D((2, 2)),
    
    # Convolutional Block 2
    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'),
    MaxPooling2D((2, 2)),

    # Convolutional Block 3
    Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'),
    MaxPooling2D((2, 2)),
    
    # Flatten and Dense layers
    Flatten(),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax') # Output layer
])

# --- Compile the Model ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Train the Model ---
print("\n--- Starting Model Training ---")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)
print("--- Model Training Finished ---")


# --- Save the Model ---
model_path = os.path.join(os.path.dirname(__file__), 'fer_model.h5')
model.save(model_path)
print(f"FER model saved successfully to {model_path}")

# Also save the class names, as they are crucial for prediction
labels_path = os.path.join(os.path.dirname(__file__), 'fer_class_names.npy')
np.save(labels_path, class_names)
print(f"Class names saved to {labels_path}")