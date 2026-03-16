from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GaussianNoise,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def create_random_forest(random_state=123):
    return RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )


def build_mobilenet_basic(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    return base_model, model


def build_mobilenet_improved(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = GaussianNoise(0.03)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.35)(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    return base_model, model


def build_mobilenet_final(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = GaussianNoise(0.1)(x)
    x = Dense(192, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.50)(x)
    x = Dense(96, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.30)(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    return base_model, model


def build_resnet50_basic(input_shape, num_classes):
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    return base_model, model


def build_simple_cnn(input_shape=(128, 128, 1), num_classes=3, dropout_rate=0.3):
    """Simple CNN trained from scratch (no transfer learning)."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout_rate),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout_rate),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout_rate),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(dropout_rate),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model
