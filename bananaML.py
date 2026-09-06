import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# 1. CREATE DATAFRAME (.csv)
path_root = "banana"
arr = []

for banana_id in os.listdir(path_root):
    banana_id_col = banana_id.replace("Banana_ID_", "")
    id_path = os.path.join(path_root, banana_id)
    
    if not os.path.isdir(id_path):
        continue
        
    for bananaDays in os.listdir(id_path):
        if bananaDays.endswith(".jpg"):
            day = int(bananaDays.replace("Day_", "").replace(".jpg", ""))
            remaining_days = 7 - day
            image_path = os.path.join(id_path, bananaDays)
            arr.append([image_path, banana_id_col, day, remaining_days])

data = pd.DataFrame(arr, columns=["image_path", "banana_id", "day", "remaining_days"])
data.to_csv("banana.csv", index=False)
print("CSV file created successfully!")

#2. SPLIT DATA
unique_ids = data["banana_id"].unique()

train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

train_data = data[data["banana_id"].isin(train_ids)]
val_data = data[data["banana_id"].isin(val_ids)]
test_data = data[data["banana_id"].isin(test_ids)]

print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")

#3. DATA GENERATORS
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

img_size = (224, 224)
batch_size = 32

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col="image_path",
    y_col="remaining_days",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="raw",
    shuffle=True
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_data,
    x_col="image_path",
    y_col="remaining_days",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="raw",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col="image_path",
    y_col="remaining_days",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="raw",
    shuffle=False
)

#4. BUILD MODEL
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='linear')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

#5. TRAIN MODEL
print("\nPhase 1: Training with frozen base model")
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

print("\nPhase 2: Fine-tuning")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='mse',
    metrics=['mae']
)

model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

#6. SAVE MODEL
model.save('banana_model.h5')
print("Model saved successfully!")

#7. EVALUATE ON TEST SET
print("\n===== Evaluation on Test Set =====")
test_loss, test_mae = model.evaluate(test_generator)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

#8. SHOW SOME PREDICTIONS
test_images, test_labels = next(test_generator)
predictions = model.predict(test_images, verbose=0)

print("\nActual vs Predicted (first 10 samples):")
print("-" * 55)
for i in range(min(10, len(test_labels))):
    true_days = test_labels[i]
    pred_days = predictions[i][0]
    print(f"Actual: {true_days:.1f} days  |  Predicted: {pred_days:.1f} days  |  Diff: {abs(true_days - pred_days):.1f}")