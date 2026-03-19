import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_sklearn_classifier(
    model,
    X_test,
    y_test,
    label_encoder,
    title,
    cmap="Blues",
):
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    target_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{'=' * 60}")
    print(f"TEST SET EVALUATION - {title}")
    print(f"{'=' * 60}\n")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
    print("Per-class Metrics (Precision / Recall / F1-score / Support):")
    print(report)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap=cmap, values_format="d")
    plt.title(f"{title} - Confusion Matrix\nTest Set Performance", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()

    return y_pred, test_accuracy, report, cm


def evaluate_keras_classifier(
    model,
    X_test,
    y_test,
    label_encoder,
    title,
    cmap="Blues",
):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    target_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"TEST SET EVALUATION - {title}")
    print(f"{'=' * 60}")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nPer-class Metrics (Precision / Recall / F1-score / Support):")
    print(report)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap=cmap, values_format="d")
    plt.title(f"{title} - Confusion Matrix\nTest Set Performance", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()

    return y_pred, accuracy, loss, report, cm


def plot_training_history(history, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"{title} Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"{title} Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()


def compare_model_accuracies(model_names, accuracies):
    comparison_df = pd.DataFrame(
        {
            "Model": model_names,
            "Test Accuracy": [f"{accuracy * 100:.2f}%" for accuracy in accuracies],
        }
    )

    print(f"\n{'=' * 60}")
    print("MODEL PERFORMANCE COMPARISON (Test Set)")
    print(f"{'=' * 60}")
    print(comparison_df.to_string(index=False))
    print(f"{'=' * 60}")

    plt.figure(figsize=(12, 7))
    sns.barplot(x=model_names, y=accuracies, palette="viridis")
    plt.title("Test Accuracy Comparison Across Models", fontsize=16, fontweight="bold")
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.ylim(0, 1)
    for index, value in enumerate(accuracies):
        plt.text(index, value + 0.02, f"{value * 100:.2f}%", color="black", ha="center")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    return comparison_df


def plot_side_by_side_confusion_matrices(cm_left, cm_right, labels, left_title, right_title):
    print("Generating side-by-side Confusion Matrices for comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    disp_left = ConfusionMatrixDisplay(confusion_matrix=cm_left, display_labels=labels)
    disp_left.plot(ax=axes[0], cmap="Greens", values_format="d")
    axes[0].set_title(left_title, fontsize=14, fontweight="bold", pad=20)
    axes[0].set_xlabel("Predicted Label", fontsize=12)
    axes[0].set_ylabel("True Label", fontsize=12)

    disp_right = ConfusionMatrixDisplay(confusion_matrix=cm_right, display_labels=labels)
    disp_right.plot(ax=axes[1], cmap="Blues", values_format="d")
    axes[1].set_title(right_title, fontsize=14, fontweight="bold", pad=20)
    axes[1].set_xlabel("Predicted Label", fontsize=12)
    axes[1].set_ylabel("True Label", fontsize=12)

    plt.suptitle("Comparison of Models on Test Set", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_all_confusion_matrices(confusion_matrices, labels, titles, cmaps):
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    for axis, cm, title, cmap in zip(axes, confusion_matrices, titles, cmaps):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=axis, cmap=cmap, values_format="d")
        axis.set_title(title, fontsize=14, fontweight="bold", pad=10)
        axis.set_xlabel("Predicted Label", fontsize=12)
        axis.set_ylabel("True Label", fontsize=12)

    plt.suptitle("Confusion Matrices Comparison on Test Set", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()


def _get_default_comparison_metrics_df():
    return pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "MobileNetV2",
                "MobileNetV2 2.0",
                "MobileNetV2 3.0",
                "ResNet50",
                "ResNet50 2.0",
            ],
            "Test Accuracy": [0.6667, 0.7143, 0.7619, 0.8095, 0.5034, 0.6395],
            "Malignant Recall": [0.6667, 0.7667, 0.7333, 0.7333, 0.5667, 0.6333],
            "Malignant Precision": [0.5263, 0.6216, 0.6286, 0.6875, 0.3696, 0.5938],
        }
    )


def plot_comparative_metrics_section(metrics_df=None):
    if metrics_df is None:
        metrics_df = _get_default_comparison_metrics_df()

    melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(13, 5))
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.title("Comparative Evaluation Metrics Across Models", fontsize=14, fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=18, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.show()


def plot_data_centric_section(namespace=None, false_normals=None, versions=None):
    if namespace is None:
        namespace = {}

    if versions is None:
        versions = ["MobileNetV2", "MobileNetV2 2.0", "MobileNetV2 3.0"]
    if false_normals is None:
        false_normals = [16, 15, 9]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(versions, false_normals, marker="o", linewidth=2.5, color="#d62728")
    axes[0].set_title("False Normal Reduction", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Model Version")
    axes[0].set_ylabel("False Normal Count")
    axes[0].grid(alpha=0.25)
    for i, value in enumerate(false_normals):
        axes[0].text(i, value + 0.15, str(value), ha="center", fontsize=9)

    y_train = namespace.get("y_train")
    y_train_augmented = namespace.get("y_train_augmented")
    label_encoder = namespace.get("label_encoder")

    if y_train is not None and y_train_augmented is not None:
        raw_counts = np.bincount(np.array(y_train).astype(int))
        aug_counts = np.bincount(np.array(y_train_augmented).astype(int))
        n_classes = max(len(raw_counts), len(aug_counts))
        raw_counts = np.pad(raw_counts, (0, n_classes - len(raw_counts)))
        aug_counts = np.pad(aug_counts, (0, n_classes - len(aug_counts)))

        x = np.arange(n_classes)
        width = 0.38
        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            class_names = list(label_encoder.classes_)
        else:
            class_names = [f"Class {i}" for i in x]

        axes[1].bar(x - width / 2, raw_counts, width=width, label="Before Aug", color="#4c72b0")
        axes[1].bar(x + width / 2, aug_counts, width=width, label="After Aug", color="#55a868")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(class_names, rotation=20, ha="right")
        axes[1].set_title("Class Distribution Shift", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Sample Count")
        axes[1].legend()
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.5,
            "Class distribution plot unavailable\n(run data preparation cells first)",
            ha="center",
            va="center",
            fontsize=10,
        )

    plt.suptitle("Data-Centric Strategies: Quantitative Impact", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.show()


def plot_random_forest_feature_importance_section(namespace=None, top_k=30):
    if namespace is None:
        namespace = {}

    rf_candidates = ["rf_model", "random_forest_model", "best_rf_model", "random_forest"]
    rf_model_obj = next((namespace[name] for name in rf_candidates if name in namespace), None)

    if rf_model_obj is None or not hasattr(rf_model_obj, "feature_importances_"):
        print("Random Forest model with feature_importances_ not found. Run RF training/evaluation cells first.")
        return

    feature_importances = np.array(rf_model_obj.feature_importances_)
    grid_size = int(np.sqrt(feature_importances.size))

    if grid_size * grid_size != feature_importances.size:
        top_k = min(top_k, feature_importances.size)
        top_indices = np.argsort(feature_importances)[-top_k:][::-1]
        plt.figure(figsize=(10, 5))
        sns.barplot(x=np.arange(top_k), y=feature_importances[top_indices], palette="mako")
        plt.title("Top Random Forest Pixel Importances", fontsize=13, fontweight="bold")
        plt.xlabel("Ranked Feature Index")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()
        return

    importance_map = feature_importances.reshape(grid_size, grid_size)
    plt.figure(figsize=(6, 6))
    sns.heatmap(importance_map, cmap="inferno", cbar=True)
    plt.title("Random Forest Feature-Importance Map", fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_false_normal_reduction(versions=None, false_normals=None):
    if versions is None:
        versions = ["MobileNetV2", "MobileNetV2 2.0", "MobileNetV2 3.0"]
    if false_normals is None:
        false_normals = [16, 15, 9]

    plt.figure(figsize=(7, 4.5))
    plt.plot(versions, false_normals, marker="o", linewidth=2.5, color="#d62728")
    plt.title("False Normal Reduction", fontsize=13, fontweight="bold")
    plt.xlabel("Model Version")
    plt.ylabel("False Normal Count")
    plt.grid(alpha=0.25)
    for i, value in enumerate(false_normals):
        plt.text(i, value + 0.15, str(value), ha="center", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_random_forest_pixel_importance_map(namespace=None, top_k=30):
    if namespace is None:
        namespace = {}

    rf_candidates = ["rf_model", "random_forest_model", "best_rf_model", "random_forest"]
    rf_model_obj = next((namespace[name] for name in rf_candidates if name in namespace), None)

    if rf_model_obj is None or not hasattr(rf_model_obj, "feature_importances_"):
        print("Random Forest model with feature_importances_ not found. Run RF training/evaluation cells first.")
        return

    feature_importances = np.array(rf_model_obj.feature_importances_)
    grid_size = int(np.sqrt(feature_importances.size))

    if grid_size * grid_size == feature_importances.size:
        importance_map = feature_importances.reshape(grid_size, grid_size)
        plt.figure(figsize=(6, 6))
        sns.heatmap(importance_map, cmap="inferno", cbar=True)
        plt.title("Random Forest Pixel-Importance Map", fontsize=13, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        return

    top_k = min(top_k, feature_importances.size)
    top_indices = np.argsort(feature_importances)[-top_k:][::-1]
    plt.figure(figsize=(10, 4.5))
    sns.barplot(x=np.arange(top_k), y=feature_importances[top_indices], palette="mako")
    plt.title("Top Random Forest Pixel Importances", fontsize=13, fontweight="bold")
    plt.xlabel("Ranked Pixel Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_random_forest_benign_misclassification_breakdown(namespace=None):
    if namespace is None:
        namespace = {}

    cm_value = namespace.get("cm")
    label_encoder = namespace.get("label_encoder")

    if cm_value is None:
        print("Confusion matrix `cm` not found. Run RF evaluation cells first.")
        return
    if label_encoder is None or not hasattr(label_encoder, "classes_"):
        print("`label_encoder` not found. Run preprocessing/encoding cells first.")
        return

    classes = list(label_encoder.classes_)
    if "benign" not in classes:
        print("Class 'benign' not found in label encoder classes.")
        return

    rf_cm = np.array(cm_value)
    benign_idx = classes.index("benign")
    benign_row = rf_cm[benign_idx]

    benign_total = int(benign_row.sum())
    benign_correct = int(benign_row[benign_idx])
    benign_misclassified = benign_total - benign_correct

    print(f"Benign total test samples: {benign_total}")
    print(f"Benign correctly classified: {benign_correct}")
    print(f"Benign misclassified: {benign_misclassified}")

    mis_labels = []
    mis_counts = []
    for j, cls in enumerate(classes):
        if j != benign_idx:
            mis_labels.append(cls)
            mis_counts.append(int(benign_row[j]))

    plt.figure(figsize=(6, 4))
    sns.barplot(x=mis_labels, y=mis_counts, palette="Set2")
    plt.title("RF Benign Misclassification Breakdown", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    for i, v in enumerate(mis_counts):
        plt.text(i, v + 0.05, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_resolution_impact_with_samples(
    namespace=None,
    sample_count=4,
    resolution_df=None,
    random_state=22,
):
    if namespace is None:
        namespace = {}

    if resolution_df is None:
        resolution_df = pd.DataFrame(
            {
                "Model": ["MobileNetV2 2.0", "MobileNetV2 3.0"],
                "Resolution": ["128x128", "224x224"],
                "Pixels": [128 * 128, 224 * 224],
                "Test Accuracy": [76.19, 80.95],
                "False Normals": [15, 9],
            }
        )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    sns.barplot(data=resolution_df, x="Resolution", y="Test Accuracy", hue="Model", palette="Set2", ax=axes[0])
    axes[0].set_title("Accuracy by Input Resolution", fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xlabel("Resolution")

    sns.barplot(data=resolution_df, x="Resolution", y="False Normals", hue="Model", palette="Set2", ax=axes[1])
    axes[1].set_title("False Normals by Resolution", fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Resolution")

    sns.barplot(data=resolution_df, x="Resolution", y="Pixels", palette="flare", ax=axes[2])
    axes[2].set_title("Input Feature Space", fontweight="bold")
    axes[2].set_ylabel("Pixels per Image")
    axes[2].set_xlabel("Resolution")
    for i, p in enumerate(resolution_df["Pixels"]):
        axes[2].text(i, p + 1000, f"{p:,}", ha="center", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, title="Model", loc="best")
    axes[1].legend([], [], frameon=False)

    plt.suptitle("Resolution Impact: 128x128 vs 224x224", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.show()

    print(f"Feature-space scale-up from 128x128 to 224x224: {(224 * 224) / (128 * 128):.2f}x")

    x_128 = namespace.get("X_test_rgb")
    x_224 = namespace.get("X_test_rgb_final224")
    y_test = namespace.get("y_test")
    label_encoder = namespace.get("label_encoder")

    if x_128 is None or x_224 is None:
        print("Sample visualization skipped: `X_test_rgb` and/or `X_test_rgb_final224` not found.")
        return

    x_128 = np.array(x_128)
    x_224 = np.array(x_224)
    n = min(len(x_128), len(x_224))
    if n == 0:
        print("Sample visualization skipped: empty test arrays.")
        return

    sample_count = max(1, min(sample_count, n))
    rng = np.random.default_rng(random_state)
    sample_indices = rng.choice(n, size=sample_count, replace=False)

    fig, axes = plt.subplots(sample_count, 2, figsize=(8, 3 * sample_count))
    if sample_count == 1:
        axes = np.array([axes])

    class_names = None
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        class_names = list(label_encoder.classes_)

    for row_idx, sample_idx in enumerate(sample_indices):
        img_128 = x_128[sample_idx]
        img_224 = x_224[sample_idx]

        img_128 = np.clip(img_128, 0.0, 1.0)
        img_224 = np.clip(img_224, 0.0, 1.0)

        axes[row_idx, 0].imshow(img_128)
        axes[row_idx, 0].set_title("128x128", fontsize=10)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(img_224)
        axes[row_idx, 1].set_title("224x224", fontsize=10)
        axes[row_idx, 1].axis("off")

        if y_test is not None and class_names is not None and sample_idx < len(y_test):
            label_idx = int(y_test[sample_idx])
            if 0 <= label_idx < len(class_names):
                axes[row_idx, 0].set_ylabel(class_names[label_idx], fontsize=10)

    plt.suptitle("Actual Test Samples: 128x128 vs 224x224", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_class_imbalance_progress(namespace=None):
    if namespace is None:
        namespace = {}

    y_train = namespace.get("y_train")
    y_train_augmented = namespace.get("y_train_augmented")
    final_y_train_augmented = namespace.get("final_y_train_augmented")
    label_encoder = namespace.get("label_encoder")

    if y_train is None:
        print("`y_train` not found. Run data split cells first.")
        return

    y_train = np.array(y_train).astype(int)
    y_train_augmented = np.array(y_train_augmented).astype(int) if y_train_augmented is not None else None
    final_y_train_augmented = (
        np.array(final_y_train_augmented).astype(int) if final_y_train_augmented is not None else None
    )

    n_classes = int(np.max(y_train)) + 1
    if y_train_augmented is not None:
        n_classes = max(n_classes, int(np.max(y_train_augmented)) + 1)
    if final_y_train_augmented is not None:
        n_classes = max(n_classes, int(np.max(final_y_train_augmented)) + 1)

    class_names = [f"Class {i}" for i in range(n_classes)]
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        classes = list(label_encoder.classes_)
        if len(classes) == n_classes:
            class_names = classes

    stage_labels = ["No Augmentation", "Base Augmentation"]
    stage_counts = [np.bincount(y_train, minlength=n_classes), np.bincount(y_train_augmented, minlength=n_classes)]

    if final_y_train_augmented is not None:
        stage_labels.append("Final Augmentation")
        stage_counts.append(np.bincount(final_y_train_augmented, minlength=n_classes))

    counts_df = pd.DataFrame(stage_counts, index=stage_labels, columns=class_names)
    pct_df = counts_df.div(counts_df.sum(axis=1), axis=0) * 100

    print("\nCLASS COUNTS BY STAGE")
    print(counts_df.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts_df.plot(kind="bar", ax=axes[0], colormap="Set2")
    axes[0].set_title("Class Counts Across Augmentation Stages", fontweight="bold")
    axes[0].set_xlabel("Training Stage")
    axes[0].set_ylabel("Sample Count")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.heatmap(pct_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Class Percentage per Stage (%)", fontweight="bold")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Training Stage")

    plt.tight_layout()
    plt.show()


def plot_training_loss_section(namespace=None, model_names=None):
    if namespace is None:
        namespace = {}

    history_candidates = [
        ("MobileNetV2 (Basic)", "history"),
        ("MobileNetV2 2.0", "improved_finetune_history"),
        ("MobileNetV2 3.0", "final_finetune_history"),
        ("ResNet50 (Basic)", "basic_resnet_history"),
        ("ResNet50 2.0", "resnet_finetune_history"),
    ]
    available_histories = [(name, namespace[var]) for name, var in history_candidates if var in namespace]

    if model_names is not None:
        selected_names = set(model_names)
        available_histories = [item for item in available_histories if item[0] in selected_names]

    if not available_histories:
        print("No training history objects found. Run the training cells first.")
        return

    n = len(available_histories)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (model_name, hist_obj) in zip(axes, available_histories):
        history_dict = hist_obj.history if hasattr(hist_obj, "history") else hist_obj
        train_loss = history_dict.get("loss", [])
        val_loss = history_dict.get("val_loss", [])

        ax.plot(train_loss, label="Train Loss", linewidth=2)
        if len(val_loss) > 0:
            ax.plot(val_loss, label="Val Loss", linewidth=2)

        ax.set_title(f"{model_name}: Training vs Validation Loss", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices_section(namespace=None, normalize=True, model_names=None):
    if namespace is None:
        namespace = {}

    cm_candidates = [
        ("Random Forest", ["cm"]),
        ("MobileNetV2", ["cm_mobilenet"]),
        ("MobileNetV2 2.0", ["cm_improved_mobilenet", "cm_improved"]),
        ("MobileNetV2 3.0", ["cm_final_mobilenet", "cm_final"]),
        ("ResNet50", ["cm_resnet"]),
        ("ResNet50 2.0", ["cm_resnet_finetuned"]),
    ]

    available_cms = []
    for model_name, cm_vars in cm_candidates:
        cm_value = next((namespace.get(var_name) for var_name in cm_vars if namespace.get(var_name) is not None), None)
        if cm_value is not None:
            cm_array = np.array(cm_value)
            if cm_array.ndim == 2:
                available_cms.append((model_name, cm_array))

    if model_names is not None:
        selected_names = set(model_names)
        available_cms = [item for item in available_cms if item[0] in selected_names]

    if not available_cms:
        print("No confusion matrices found. Run model evaluation cells first.")
        return

    label_encoder = namespace.get("label_encoder")
    class_labels = list(label_encoder.classes_) if label_encoder is not None and hasattr(label_encoder, "classes_") else "auto"

    n = len(available_cms)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (model_name, cm_value) in enumerate(available_cms):
        ax = axes[i]
        plot_matrix = cm_value
        fmt = "d"
        vmin, vmax = None, None
        title_suffix = "CM"

        if normalize:
            row_sums = cm_value.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            plot_matrix = cm_value / row_sums
            fmt = ".2f"
            vmin, vmax = 0, 1
            title_suffix = "Normalized CM"

        sns.heatmap(
            plot_matrix,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
        )
        ax.set_title(f"{model_name}\n{title_suffix}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Error Analysis: Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_model_complexity_section(complexity_df=None):
    if complexity_df is None:
        complexity_df = pd.DataFrame(
            {
                "Model Family": ["Random Forest", "MobileNetV2", "ResNet50"],
                "Approx Params (M)": [0.0, 2.2, 25.0],
                "Representative Test Accuracy": [0.6667, 0.7687, 0.6395],
            }
        )

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=complexity_df,
        x="Approx Params (M)",
        y="Representative Test Accuracy",
        hue="Model Family",
        s=180,
        palette="Set1",
    )

    for _, row in complexity_df.iterrows():
        plt.text(
            row["Approx Params (M)"] + 0.2,
            row["Representative Test Accuracy"] + 0.005,
            row["Model Family"],
            fontsize=9,
        )

    plt.title("Model Complexity vs Performance", fontsize=13, fontweight="bold")
    plt.xlabel("Approximate Trainable Parameters (Millions)")
    plt.ylabel("Test Accuracy")
    plt.ylim(0.45, 0.82)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_resolution_impact_section(resolution_df=None):
    if resolution_df is None:
        resolution_df = pd.DataFrame(
            {
                "Setting": ["MobileNetV2 2.0 (128x128)", "MobileNetV2 3.0 (224x224)"],
                "Resolution": [128, 224],
                "False Normals": [15, 13],
            }
        )

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax2 = ax1.twinx()

    x = range(len(resolution_df))
    ax1.bar(x, resolution_df["Resolution"], color="#4c72b0", alpha=0.7, label="Input Resolution")
    ax2.plot(x, resolution_df["False Normals"], color="#c44e52", marker="o", linewidth=2.5, label="False Normals")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(resolution_df["Setting"], rotation=15, ha="right")
    ax1.set_ylabel("Resolution (pixels)")
    ax2.set_ylabel("False Normal Count")
    ax1.set_title("Feature Extraction Perspective: More Pixels, Fewer Missed Cases", fontsize=12, fontweight="bold")

    for i, value in enumerate(resolution_df["False Normals"]):
        ax2.text(i, value + 0.1, str(value), color="#c44e52", ha="center", fontsize=9)

    fig.tight_layout()
    plt.show()


def plot_precision_recall_tradeoff_section(pr_tradeoff_df=None):
    if pr_tradeoff_df is None:
        pr_tradeoff_df = pd.DataFrame(
            {
                "Model": [
                    "Random Forest",
                    "MobileNetV2",
                    "MobileNetV2 2.0",
                    "MobileNetV2 3.0",
                    "ResNet50",
                    "ResNet50 2.0",
                ],
                "Malignant Recall": [0.6667, 0.7667, 0.7333, 0.7333, 0.5667, 0.6333],
                "Malignant Precision": [0.5263, 0.6216, 0.6286, 0.6286, 0.3696, 0.5938],
            }
        )

    plt.figure(figsize=(8.5, 5))
    sns.scatterplot(
        data=pr_tradeoff_df,
        x="Malignant Recall",
        y="Malignant Precision",
        hue="Model",
        s=140,
        palette="tab10",
    )

    for _, row in pr_tradeoff_df.iterrows():
        plt.text(row["Malignant Recall"] + 0.003, row["Malignant Precision"] + 0.003, row["Model"], fontsize=8)

    plt.title("Precision-Recall Trade-off (Malignant Class)", fontsize=13, fontweight="bold")
    plt.xlim(0.5, 0.8)
    plt.ylim(0.35, 0.67)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(alpha=0.25)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()


def plot_final_evaluation_dashboard_section(namespace=None):
    if namespace is None:
        namespace = {}

    results_catalog = [
        ("Random Forest", ["cm"], ["test_accuracy"], ["y_test_pred"]),
        ("MobileNetV2 (Basic)", ["cm_mobilenet"], ["accuracy"], ["y_pred_mobilenet"]),
        (
            "MobileNetV2 2.0",
            ["cm_improved_mobilenet", "cm_improved"],
            ["improved_accuracy"],
            ["y_pred_improved_mobilenet"],
        ),
        (
            "MobileNetV2 3.0",
            ["cm_final_mobilenet", "cm_final"],
            ["final_accuracy"],
            ["y_pred_final_mobilenet"],
        ),
        ("ResNet50 (Basic)", ["cm_resnet"], ["resnet_accuracy"], ["y_pred_resnet"]),
        (
            "ResNet50 2.0",
            ["cm_resnet_finetuned"],
            ["resnet_finetuned_accuracy"],
            ["y_pred_resnet_finetuned"],
        ),
    ]

    available = []
    for model_name, cm_vars, acc_vars, pred_vars in results_catalog:
        cm_value = next((namespace.get(var_name) for var_name in cm_vars if namespace.get(var_name) is not None), None)
        acc_value = next((namespace.get(var_name) for var_name in acc_vars if namespace.get(var_name) is not None), None)
        y_pred = next((namespace.get(var_name) for var_name in pred_vars if namespace.get(var_name) is not None), None)
        if cm_value is not None and hasattr(cm_value, "shape") and np.array(cm_value).ndim == 2:
            available.append((model_name, np.array(cm_value), acc_value, y_pred))

    if not available:
        raise ValueError("No confusion matrices found. Run the model evaluation cells first, then rerun this cell.")

    label_encoder = namespace.get("label_encoder")
    label_names = list(label_encoder.classes_) if label_encoder is not None and hasattr(label_encoder, "classes_") else None
    y_test = namespace.get("y_test")

    metrics_rows = []
    malignant_label = None
    if label_names is not None and "malignant" in label_names:
        malignant_label = label_names.index("malignant")

    for model_name, cm_value, acc_value, y_pred in available:
        row = {
            "Model": model_name,
            "Accuracy": np.nan,
            "Macro Precision": np.nan,
            "Macro Recall": np.nan,
            "Macro F1": np.nan,
            "Malignant Precision": np.nan,
            "Malignant Recall": np.nan,
            "Malignant F1": np.nan,
        }

        if y_test is not None and y_pred is not None:
            y_true_arr = np.array(y_test)
            y_pred_arr = np.array(y_pred)
            if y_true_arr.shape[0] == y_pred_arr.shape[0]:
                report_dict = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)
                row["Accuracy"] = float(report_dict.get("accuracy", np.nan))
                row["Macro Precision"] = float(report_dict.get("macro avg", {}).get("precision", np.nan))
                row["Macro Recall"] = float(report_dict.get("macro avg", {}).get("recall", np.nan))
                row["Macro F1"] = float(report_dict.get("macro avg", {}).get("f1-score", np.nan))

                if malignant_label is not None:
                    malignant_key = str(malignant_label)
                    malignant_stats = report_dict.get(malignant_key, {})
                    row["Malignant Precision"] = float(malignant_stats.get("precision", np.nan))
                    row["Malignant Recall"] = float(malignant_stats.get("recall", np.nan))
                    row["Malignant F1"] = float(malignant_stats.get("f1-score", np.nan))

        if np.isnan(row["Accuracy"]) and acc_value is not None:
            row["Accuracy"] = float(acc_value)

        metrics_rows.append(row)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        numeric_cols = [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Malignant Precision",
            "Malignant Recall",
            "Malignant F1",
        ]
        for col in numeric_cols:
            metrics_df[col] = metrics_df[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "N/A")

        print("\nPER-MODEL METRICS SUMMARY")
        print(metrics_df.to_string(index=False))

    acc_models = [item[0] for item in available if item[2] is not None]
    acc_values = [float(item[2]) for item in available if item[2] is not None]

    if acc_values:
        order = np.argsort(acc_values)[::-1]
        acc_models = [acc_models[i] for i in order]
        acc_values = [acc_values[i] for i in order]

        plt.figure(figsize=(12, 5))
        sns.barplot(x=acc_models, y=acc_values, palette="crest")
        plt.title("Final Test Accuracy Comparison", fontsize=14, fontweight="bold")
        plt.ylabel("Accuracy")
        plt.xlabel("Model")
        plt.ylim(0, 1)
        plt.xticks(rotation=18, ha="right")
        for i, value in enumerate(acc_values):
            plt.text(i, value + 0.015, f"{value * 100:.2f}%", ha="center", fontsize=9)
        plt.tight_layout()
        plt.show()

    n = len(available)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    for idx, (model_name, cm_value, _, _) in enumerate(available):
        ax = axes[idx]
        sns.heatmap(
            cm_value,
            ax=ax,
            cmap="Blues",
            annot=True,
            fmt="d",
            xticklabels=label_names,
            yticklabels=label_names,
            cbar=False,
        )
        ax.set_title(f"{model_name}\nConfusion Matrix (Counts)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Final Evaluation Dashboard", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_two_phase_training_history(
    phase1_history,
    phase2_history,
    title,
    phase1_label="Phase 1",
    phase2_label="Phase 2",
):
    """Plot a single continuous training/validation curve across two fit phases."""
    phase1 = phase1_history.history if hasattr(phase1_history, "history") else phase1_history
    phase2 = phase2_history.history if hasattr(phase2_history, "history") else phase2_history

    acc = list(phase1.get("accuracy", [])) + list(phase2.get("accuracy", []))
    val_acc = list(phase1.get("val_accuracy", [])) + list(phase2.get("val_accuracy", []))
    loss = list(phase1.get("loss", [])) + list(phase2.get("loss", []))
    val_loss = list(phase1.get("val_loss", [])) + list(phase2.get("val_loss", []))

    phase_boundary = len(phase1.get("loss", []))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train")
    if len(val_acc) > 0:
        plt.plot(val_acc, label="Validation")
    if phase_boundary > 0:
        plt.axvline(phase_boundary - 1, linestyle="--", linewidth=1, color="gray", label=f"{phase1_label} -> {phase2_label}")
    plt.title(f"{title} Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train")
    if len(val_loss) > 0:
        plt.plot(val_loss, label="Validation")
    if phase_boundary > 0:
        plt.axvline(phase_boundary - 1, linestyle="--", linewidth=1, color="gray", label=f"{phase1_label} -> {phase2_label}")
    plt.title(f"{title} Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()
