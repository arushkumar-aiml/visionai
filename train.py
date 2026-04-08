"""
VisionAI — Model Training Script
Trains a TensorFlow/Keras image classifier on custom data in data/train/
Auto-detects class names from subfolders.

Built by Arush Kumar & Ayushi Shukla | github.com/arushkumar-aiml/visionai
Usage: python train.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

layers = keras.layers
models = keras.models
MobileNetV2 = keras.applications.MobileNetV2

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_DIR = "data/train"
VAL_DIR = "data/test"
MODEL_PATH = "models/visionai_model.keras"   # .keras format (recommended)
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


def check_data_directory():
    if not os.path.exists(TRAIN_DIR):
        print(f"❌  Training directory not found: {TRAIN_DIR}")
        raise SystemExit(1)

    classes = [
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d)) and not d.startswith(".")
    ]

    if len(classes) < 2:
        print(f"❌  Found only {len(classes)} class(es). Need at least 2.")
        raise SystemExit(1)

    print(f"✅  Found {len(classes)} classes: {', '.join(sorted(classes))}")
    return sorted(classes)


def load_datasets(class_names):
    print(f"\n📂  Loading training data from {TRAIN_DIR} ...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=class_names,
    )

    use_separate_val = (
        os.path.exists(VAL_DIR)
        and any(os.path.isdir(os.path.join(VAL_DIR, c)) for c in class_names)
    )

    if use_separate_val:
        print(f"📂  Loading validation data from {VAL_DIR} ...")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            VAL_DIR,
            seed=SEED,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            class_names=class_names,
        )
    else:
        print("📂  Using 20% of training data for validation.")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR,
            validation_split=0.2,
            subset="validation",
            seed=SEED,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            class_names=class_names,
        )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


def build_model(num_classes: int) -> tf.keras.Model:
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="VisionAI")


def train():
    print("=" * 60)
    print("  🔭  VisionAI — Model Training")
    print("  Built by Arush Kumar & Ayushi Shukla")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    class_names = check_data_directory()
    num_classes = len(class_names)

    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)
    print(f"💾  Class names saved to {CLASS_NAMES_PATH}")

    train_ds, val_ds = load_datasets(class_names)

    print(f"\n🧠  Building MobileNetV2 transfer-learning model ...")
    model = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # ── FIX: Only ModelCheckpoint used
    # EarlyStopping + ReduceLROnPlateau removed — Python 3.13 deepcopy bug
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\n🚀  Training for {EPOCHS} epochs ...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    best_val_acc = max(history.history.get("val_accuracy", [0])) * 100
    final_train_acc = history.history["accuracy"][-1] * 100

    print("\n" + "=" * 60)
    print(f"✅  Training complete!")
    print(f"    Best validation accuracy : {best_val_acc:.1f}%")
    print(f"    Final training accuracy  : {final_train_acc:.1f}%")
    print(f"    Model saved to           : {MODEL_PATH}")
    print(f"    Classes                  : {', '.join(class_names)}")
    print("=" * 60)
    print("\n▶  Now run: streamlit run app.py")

    # ── Fine-tuning (only if accuracy is low) ─────────────────────
    if best_val_acc < 85 and num_classes <= 10:
        print("\n🔧  Accuracy < 85% — running fine-tuning phase ...")
        try:
            base_model_layer = model.get_layer("mobilenetv2_1.00_224")
            base_model_layer.trainable = True
            for layer in base_model_layer.layers[:-30]:
                layer.trainable = False

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            ft_callbacks = [
                keras.callbacks.ModelCheckpoint(
                    MODEL_PATH,
                    monitor="val_accuracy",
                    save_best_only=True,
                    verbose=1,
                ),
            ]

            ft_history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=5,
                callbacks=ft_callbacks,
            )

            fine_val_acc = max(ft_history.history.get("val_accuracy", [0])) * 100
            print(f"✅  Fine-tuning complete! Best val accuracy: {fine_val_acc:.1f}%")

        except Exception as e:
            print(f"⚠️  Fine-tuning skipped: {e}")


if __name__ == "__main__":
    train()