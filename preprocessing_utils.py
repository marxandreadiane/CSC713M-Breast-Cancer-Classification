import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def preprocess_images(df_all_images, target_size=(128, 128), clip_limit=2.0, tile_grid_size=(8, 8)):
    images = []
    labels = []
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    print(f"Preprocessing {len(df_all_images)} images...")

    for _, row in tqdm(df_all_images.iterrows(), total=len(df_all_images)):
        try:
            img = cv2.imread(row["image_path"])
            if img is None:
                raise ValueError("Image could not be read.")

            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            img = clahe.apply(img)
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(row["label"])
        except Exception as error:
            print(f"Error processing {row['image_path']}: {error}")

    X = np.array(images)
    y = np.array(labels)
    X = np.expand_dims(X, axis=-1)
    return X, y


def encode_labels(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return label_encoder, y_encoded


def split_dataset(X, y_encoded, random_state=123):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=random_state,
        stratify=y_encoded,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def summarize_split(X, X_train, X_val, X_test, y_train, y_val, y_test, label_encoder):
    print(f"\n{'=' * 60}")
    print("Dataset Split Summary")
    print(f"{'=' * 60}")
    print(f"Training set:   {len(X_train)} images ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"Validation set: {len(X_val)} images ({len(X_val) / len(X) * 100:.1f}%)")
    print(f"Test set:       {len(X_test)} images ({len(X_test) / len(X) * 100:.1f}%)")
    print(f"{'=' * 60}")

    split_targets = [
        ("Training", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ]
    for split_name, split_values in split_targets:
        print(f"\nClass distribution in {split_name} set:")
        unique_labels, counts = np.unique(split_values, return_counts=True)
        for label_idx, count in zip(unique_labels, counts):
            label_name = label_encoder.classes_[label_idx]
            print(f"  {label_name}: {count} ({count / len(split_values) * 100:.2f}%)")


def augment_target_class(
    X_train,
    y_train,
    target_label,
    augmentation_factor=1,
    transforms=None,
    rotation_angle=15,
):
    augmented_images = []
    augmented_labels = []

    if transforms is None:
        transforms = ["flip"]

    target_indices = np.where(y_train == target_label)[0]
    print(f"Augmenting {len(target_indices)} images from target class...")

    num_augs = max(0, int(augmentation_factor))
    active_transforms = transforms if transforms else ["flip"]

    for idx in tqdm(target_indices):
        img = X_train[idx]
        img_2d = img[:, :, 0] if img.shape[-1] == 1 else img

        augmented_images.append(img)
        augmented_labels.append(y_train[idx])

        for aug_idx in range(num_augs):
            transform_name = active_transforms[aug_idx % len(active_transforms)]
            if transform_name == "flip":
                transformed = cv2.flip(img_2d, 1)
            elif transform_name == "rotate":
                h, w = img_2d.shape[:2]
                center = (w // 2, h // 2)
                signed_angle = rotation_angle if aug_idx % 2 == 0 else -rotation_angle
                rotation_matrix = cv2.getRotationMatrix2D(center, signed_angle, 1.0)
                transformed = cv2.warpAffine(
                    img_2d,
                    rotation_matrix,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
            else:
                continue

            transformed = np.expand_dims(transformed, axis=-1)
            augmented_images.append(transformed)
            augmented_labels.append(y_train[idx])

    non_target_indices = np.where(y_train != target_label)[0]
    for idx in non_target_indices:
        augmented_images.append(X_train[idx])
        augmented_labels.append(y_train[idx])

    return np.array(augmented_images), np.array(augmented_labels)


def summarize_augmentation(X_train, X_train_augmented, y_train, y_train_augmented, label_encoder, target_label):
    print(f"\n{'=' * 60}")
    print("Augmentation Results")
    print(f"{'=' * 60}")
    print(f"Training set after augmentation: {len(X_train_augmented)} images")
    print(f"Increase: +{len(X_train_augmented) - len(X_train)} images")
    print("\nClass distribution after augmentation:")

    unique_aug, counts_aug = np.unique(y_train_augmented, return_counts=True)
    for label_idx, count in zip(unique_aug, counts_aug):
        label_name = label_encoder.classes_[label_idx]
        print(f"  {label_name}: {count} ({count / len(y_train_augmented) * 100:.2f}%)")


def flatten_images(X_train_augmented, X_val, X_test):
    X_train_flat = X_train_augmented.reshape(X_train_augmented.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    return X_train_flat, X_val_flat, X_test_flat


def convert_grayscale_to_rgb(X_train_augmented, X_val, X_test):
    X_train_rgb = np.repeat(X_train_augmented, 3, axis=-1)
    X_val_rgb = np.repeat(X_val, 3, axis=-1)
    X_test_rgb = np.repeat(X_test, 3, axis=-1)
    return X_train_rgb, X_val_rgb, X_test_rgb
