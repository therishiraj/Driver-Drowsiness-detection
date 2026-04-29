"""
CNNClassifier — REASONING sub-module
Custom lightweight CNN for binary eye-state classification (open/closed).
Trained on MRL Eye Dataset augmented with brightness/contrast/occlusion transforms.
Falls back to EAR-only when model file is missing (demo mode).
"""

import cv2
import numpy as np
import os
import pickle


class CNNClassifier:
    """
    Lightweight CNN eye-state classifier.

    Architecture (designed to run < 5ms on CPU):
      Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
      Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
      Conv2D(64, 3×3, ReLU)
      Flatten → Dense(128, ReLU) → Dropout(0.5)
      Dense(1, Sigmoid) → {open, closed}

    Input: 64×32 grayscale eye crop
    Output: (label: str, confidence: float)
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'eye_cnn.pkl')
    INPUT_SHAPE = (32, 64)  # H × W

    def __init__(self):
        self._model = None
        self._using_keras = False
        self._load_model()

    def _load_model(self):
        """Load trained model; gracefully degrade if unavailable."""
        # Try Keras/TF first
        try:
            import tensorflow as tf
            keras_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'eye_cnn.h5')
            if os.path.exists(keras_path):
                self._model = tf.keras.models.load_model(keras_path)
                self._using_keras = True
                return
        except Exception:
            pass

        # Try pickled sklearn/numpy model
        if os.path.exists(self.MODEL_PATH):
            try:
                with open(self.MODEL_PATH, 'rb') as f:
                    self._model = pickle.load(f)
                return
            except Exception:
                pass

        # No model file — use pure EAR (demo mode)
        self._model = None

    def predict(self, left_eye: np.ndarray, right_eye: np.ndarray = None):
        """
        Classify eye state from one or both eye crops.
        Returns (label, confidence) where label ∈ {'open', 'closed'}.
        """
        if self._model is None:
            # Demo mode: return neutral open to let EAR be sole decision-maker
            return 'open', 0.5

        # Use left eye preferentially, fall back to right
        eye = left_eye if left_eye is not None else right_eye
        if eye is None:
            return 'open', 0.5

        try:
            if self._using_keras:
                return self._predict_keras(eye)
            else:
                return self._predict_sklearn(eye)
        except Exception:
            return 'open', 0.5

    def _predict_keras(self, eye: np.ndarray):
        inp = self._preprocess(eye)
        inp = inp[np.newaxis, ..., np.newaxis]  # (1, 32, 64, 1)
        prob = float(self._model.predict(inp, verbose=0)[0][0])
        label = 'closed' if prob > 0.5 else 'open'
        conf = prob if label == 'closed' else 1.0 - prob
        return label, conf

    def _predict_sklearn(self, eye: np.ndarray):
        inp = self._preprocess(eye).flatten().reshape(1, -1)
        prob = self._model.predict_proba(inp)[0]
        label = 'closed' if prob[1] > 0.5 else 'open'
        conf = float(max(prob))
        return label, conf

    def _preprocess(self, eye: np.ndarray) -> np.ndarray:
        """Resize, normalise to [0, 1]."""
        resized = cv2.resize(eye, (self.INPUT_SHAPE[1], self.INPUT_SHAPE[0]))
        return resized.astype(np.float32) / 255.0


# ── Training script (run once to generate model/eye_cnn.h5) ────────────────

def train_model(data_dir: str, output_path: str = None):
    """
    Train the CNN on eye image dataset.
    Expected structure:
      data_dir/
        open/   *.jpg / *.png
        closed/ *.jpg / *.png

    Run: python -c "from utils.cnn_classifier import train_model; train_model('data/eyes')"
    """
    import tensorflow as tf
    from tensorflow import keras

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'eye_cnn.h5')

    # ── Data loading ────────────────────────────────────────────────────────
    images, labels = [], []
    for label_name, label_val in [('open', 0), ('closed', 1)]:
        folder = os.path.join(data_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 32))
            images.append(img)
            labels.append(label_val)

    X = np.array(images, dtype=np.float32) / 255.0
    X = X[..., np.newaxis]  # (N, 32, 64, 1)
    y = np.array(labels, dtype=np.float32)
    print(f"Loaded {len(X)} images  |  open={sum(y==0)}  closed={sum(y==1)}")

    # ── Augmentation ────────────────────────────────────────────────────────
    aug = keras.Sequential([
        keras.layers.RandomBrightness(0.3),
        keras.layers.RandomContrast(0.3),
    ])

    # ── Model ───────────────────────────────────────────────────────────────
    model = keras.Sequential([
        keras.layers.Input(shape=(32, 64, 1)),
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.GlobalAveragePooling2D(),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X, y,
        epochs=30,
        batch_size=32,
        validation_split=0.15,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        ]
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"Model saved to {output_path}")
    return model


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/eyes'
    train_model(data_dir)
