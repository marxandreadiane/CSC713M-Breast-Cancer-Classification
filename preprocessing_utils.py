import cv2
import numpy as np
from tqdm import tqdm


def augment_target_class(
    X_train,
    y_train,
    target_label,
    augmentation_factor=1,
    transforms=None,
    rotation_angle=15,
):
    # Create class-targeted augmented copies while keeping non-target samples unchanged.
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
        rotate_count = 0

        augmented_images.append(img)
        augmented_labels.append(y_train[idx])

        for aug_idx in range(num_augs):
            transform_name = active_transforms[aug_idx % len(active_transforms)]
            if transform_name in ("flip", "flip_h"):
                transformed = cv2.flip(img_2d, 1)
            elif transform_name == "flip_v":
                transformed = cv2.flip(img_2d, 0)
            elif transform_name == "flip_both":
                transformed = cv2.flip(img_2d, -1)
            elif transform_name in ("rotate", "rotate_alt"):
                h, w = img_2d.shape[:2]
                center = (w // 2, h // 2)
                signed_angle = rotation_angle if rotate_count % 2 == 0 else -rotation_angle
                rotate_count += 1
                rotation_matrix = cv2.getRotationMatrix2D(center, signed_angle, 1.0)
                transformed = cv2.warpAffine(
                    img_2d,
                    rotation_matrix,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
            elif transform_name in ("rotate_pos", "rotate_plus"):
                h, w = img_2d.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                transformed = cv2.warpAffine(
                    img_2d,
                    rotation_matrix,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
            elif transform_name in ("rotate_neg", "rotate_minus"):
                h, w = img_2d.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
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
    # Expand single-channel grayscale tensors to 3 channels for ImageNet backbones.
    X_train_rgb = np.repeat(X_train_augmented, 3, axis=-1)
    X_val_rgb = np.repeat(X_val, 3, axis=-1)
    X_test_rgb = np.repeat(X_test, 3, axis=-1)
    return X_train_rgb, X_val_rgb, X_test_rgb
