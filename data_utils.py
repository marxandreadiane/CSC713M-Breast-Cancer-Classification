from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def get_dataset_paths(start_dir=None):
    notebook_dir = Path(start_dir or Path.cwd())
    datasets_dir = notebook_dir / "Datasets"

    if not datasets_dir.exists():
        datasets_dir = notebook_dir / "MCO" / "Datasets"

    busi_root = datasets_dir / "Dataset_BUSI" / "Dataset_BUSI_with_GT"
    bus_uclm_root = (
        datasets_dir
        / "BUS-UCLM Breast ultrasound lesion segmentation dataset"
        / "BUS-UCLM Breast ultrasound lesion segmentation dataset"
        / "BUS-UCLM"
    )

    return datasets_dir, busi_root, bus_uclm_root


def get_label_from_mask(mask_path):
    try:
        mask = Image.open(mask_path).convert("RGB")
        mask_array = np.array(mask)

        red_pixels = np.sum(
            (mask_array[:, :, 0] > 200)
            & (mask_array[:, :, 1] < 50)
            & (mask_array[:, :, 2] < 50)
        )
        green_pixels = np.sum(
            (mask_array[:, :, 0] < 50)
            & (mask_array[:, :, 1] > 200)
            & (mask_array[:, :, 2] < 50)
        )

        if red_pixels > 100:
            return "malignant"
        if green_pixels > 100:
            return "benign"
        return "normal"
    except Exception as error:
        print(f"Error reading mask {mask_path}: {error}")
        return None


def load_busi_dataset(dataset_root):
    image_data = []

    for root, _, files in Path(dataset_root).walk():
        root_path = Path(root)

        for filename in files:
            if not filename.lower().endswith((".png")):
                continue

            file_path = root_path / filename
            label = None
            for part in root_path.parts:
                part_lower = part.lower()
                if "benign" in part_lower:
                    label = "benign"
                    break
                if "malignant" in part_lower:
                    label = "malignant"
                    break
                if "normal" in part_lower:
                    label = "normal"
                    break

            if "mask" not in filename.lower():
                image_data.append({"image_path": str(file_path), "label": label})

    return pd.DataFrame(image_data)


def load_bus_uclm_dataset(dataset_root):
    images_dir = Path(dataset_root) / "images"
    masks_dir = Path(dataset_root) / "masks"

    image_data = []

    for img_file in images_dir.glob("*.png"):
        mask_file = masks_dir / img_file.name

        if mask_file.exists():
            label = get_label_from_mask(mask_file)
            image_data.append({"image_path": str(img_file), "label": label})
        else:
            image_data.append({"image_path": str(img_file), "label": None})

    return pd.DataFrame(image_data)


def merge_image_and_mask_datasets(df_busi_images, df_bus_uclm_images):
    df_all_images = pd.concat([df_busi_images, df_bus_uclm_images], ignore_index=True)
    df_all_images = df_all_images[df_all_images["label"].notna()]
    return df_all_images
