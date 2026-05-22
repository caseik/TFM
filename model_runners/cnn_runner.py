import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, DenseNet121, InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import os


class CNNRunner:
    def __init__(self, model_name="resnet50", num_classes=7, epochs=10):
        self.model_name = model_name
        self.num_classes = num_classes
        self.epochs = epochs
        self.model = None
        self.base_model = None

    def build_model(self, base_model):
        base_model.trainable = False

        inputs = keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def get_base_model(self):
        if self.model_name == "resnet50":
            return ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        elif self.model_name == "densenet121":
            return DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        elif self.model_name == "inceptionv3":
            return InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def fine_tune(self, train_gen, val_gen, class_weights):
        self.base_model.trainable = True
        fine_tune_at = len(self.base_model.layers) - 30

        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False

        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.epochs,
            class_weight=class_weights,
            verbose=1
        )

        return history

    def run(self, split_bundle):
        train_gen = split_bundle.get("train_gen")
        val_gen = split_bundle.get("val_gen")
        test_gen = split_bundle.get("test_gen")

        self.base_model = self.get_base_model()
        self.model = self.build_model(self.base_model)

        if train_gen is not None:
            y_train = train_gen.classes
            weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights = dict(enumerate(weights))

            self.fine_tune(train_gen, val_gen, class_weights)

        if test_gen is not None:
            y_true = test_gen.classes
            y_pred = np.argmax(self.model.predict(test_gen), axis=1)

            return {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_pred_proba": self.model.predict(test_gen),
                "class_names": list(test_gen.class_indices.keys()),
                "model": self.model,
                "history": None
            }

        return {}
