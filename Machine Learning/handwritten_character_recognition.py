# handwritten_character_recognition.py

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# 1. Load EMNIST Letters dataset (A–Z, 26 classes)
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# 2. Normalize & batch datasets
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)  # Add channel dimension
    label = label - 1  # EMNIST labels are 1–26; shift to 0–25
    return image, label

BATCH_SIZE = 128

ds_train = ds_train.map(preprocess).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 3. Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(26, activation="softmax")  # 26 classes: A–Z
])

# 4. Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train model
model.fit(ds_train, epochs=5, validation_data=ds_test)

# 6. Evaluate model
test_loss, test_acc = model.evaluate(ds_test)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# 7. Save model
model.save("emnist_character_model.h5")
print("💾 Model saved as 'emnist_character_model.h5'")

# 8. Visualize Predictions
class_names = [chr(i) for i in range(65, 91)]  # A–Z

# Get test data batch
for images, labels in ds_test.take(1):
    preds = model.predict(images)
    for i in range(5):
        plt.imshow(tf.squeeze(images[i]), cmap='gray')
        pred_label = class_names[np.argmax(preds[i])]
        actual_label = class_names[labels[i].numpy()]
        plt.title(f"Predicted: {pred_label} | Actual: {actual_label}")
        plt.axis("off")
        plt.show()

