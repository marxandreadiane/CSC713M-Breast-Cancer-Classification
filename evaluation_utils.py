import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix


def evaluate_sklearn_classifier(model, X_test, y_test, label_encoder, title, cmap="Blues"):
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    target_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{'=' * 60}")
    print(f"TEST SET EVALUATION - {title}")
    print(f"{'=' * 60}\n")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print("=" * 60)
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap=cmap, values_format="d")
    plt.title(f"{title} - Confusion Matrix\nTest Set Performance", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"\n{'=' * 60}")
    print("Per-Class Performance Summary:")
    print(f"{'=' * 60}")
    for index, class_name in enumerate(target_names):
        class_mask = y_test == index
        class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
        class_count = np.sum(class_mask)
        print(f"{class_name.capitalize():12s}: {class_acc * 100:6.2f}% accuracy ({class_count} samples)")
    print(f"{'=' * 60}")

    return y_pred, test_accuracy, report, cm


def evaluate_keras_classifier(model, X_test, y_test, label_encoder, title, cmap="Blues"):
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
    print("\nClassification Report:")
    print(f"{'=' * 60}")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap=cmap, values_format="d")
    plt.title(f"{title} - Confusion Matrix\nTest Set Performance", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"\n{'=' * 60}")
    print("Per-Class Performance Summary:")
    print(f"{'=' * 60}")
    for index, class_name in enumerate(target_names):
        class_mask = y_test == index
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
            class_count = np.sum(class_mask)
            print(f"{class_name.capitalize():12s}: {class_acc * 100:6.2f}% accuracy ({class_count} samples)")
        else:
            print(f"{class_name.capitalize():12s}: No samples in test set")
    print(f"{'=' * 60}")

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
    display(comparison_df)
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