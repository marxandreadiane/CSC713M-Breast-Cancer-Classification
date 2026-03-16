from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
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
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    return base_model, model


def build_mobilenet_final(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
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
