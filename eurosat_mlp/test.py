import numpy as np
import os
import json

from model import ThreeLayerMLP
from data_loader import prepare_data, CLASS_NAMES
from train import compute_accuracy


def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm, class_names):
    header = f"{'':>20s} | " + " | ".join(f"{name[:8]:>8s}" for name in class_names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = f"{name:>20s} | " + " | ".join(f"{cm[i, j]:>8d}" for j in range(len(class_names)))
        print(row)


def get_misclassified(model, X, y, paths, batch_size=512):
    n = X.shape[0]
    all_preds = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        preds = model.predict(X[start:end])
        all_preds.append(preds)
    all_preds = np.concatenate(all_preds)

    mask = all_preds != y
    return {
        'indices': np.where(mask)[0],
        'true_labels': y[mask],
        'pred_labels': all_preds[mask],
        'paths': paths[mask] if paths is not None else None,
    }


def evaluate(data_dir, model_path, save_dir='results', seed=42):
    os.makedirs(save_dir, exist_ok=True)

    data = prepare_data(data_dir, seed=seed)
    test_X, test_y, test_paths = data['test_X'], data['test_y'], data['test_paths']

    model = ThreeLayerMLP.load(model_path)

    test_acc = compute_accuracy(model, test_X, test_y)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    n = test_X.shape[0]
    all_preds = []
    for start in range(0, n, 512):
        end = min(start + 512, n)
        preds = model.predict(test_X[start:end])
        all_preds.append(preds)
    all_preds = np.concatenate(all_preds)

    cm = confusion_matrix(test_y, all_preds)
    print("\nConfusion Matrix:")
    print_confusion_matrix(cm, CLASS_NAMES)

    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    print("\nPer-class Accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:>25s}: {per_class_acc[i]:.4f}")

    misclassified = get_misclassified(model, test_X, test_y, test_paths)
    print(f"\nTotal misclassified: {len(misclassified['indices'])} / {len(test_y)}")

    results = {
        'test_acc': float(test_acc),
        'confusion_matrix': cm.tolist(),
        'per_class_acc': {name: float(per_class_acc[i]) for i, name in enumerate(CLASS_NAMES)},
        'num_misclassified': int(len(misclassified['indices'])),
    }

    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    np.savez(os.path.join(save_dir, 'misclassified.npz'),
             indices=misclassified['indices'],
             true_labels=misclassified['true_labels'],
             pred_labels=misclassified['pred_labels'],
             paths=misclassified['paths'] if misclassified['paths'] is not None else np.array([]))

    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), cm)

    print(f"\nResults saved to {save_dir}/")
    return results, cm, misclassified


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../EuroSAT_RGB')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.npz')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    evaluate(args.data_dir, args.model_path, args.save_dir, args.seed)