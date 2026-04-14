import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import CLASS_NAMES
from model import ThreeLayerMLP


def plot_training_curves(history_path, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['val_acc'], label='Val Accuracy', linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['lr'], label='Learning Rate', linewidth=2, color='red')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


def plot_confusion_matrix(cm_path_or_array, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(cm_path_or_array, str):
        cm = np.load(cm_path_or_array)
    else:
        cm = cm_path_or_array

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(CLASS_NAMES))
    short_names = [n[:10] for n in CLASS_NAMES]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(short_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=8)

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_dir}/confusion_matrix.png")


def visualize_first_layer_weights(model_path, image_size=64, save_dir='figures', n_show=36):
    os.makedirs(save_dir, exist_ok=True)

    model = ThreeLayerMLP.load(model_path)
    W1 = model.W1

    n_neurons = W1.shape[1]
    n_show = min(n_show, n_neurons)

    norms = np.linalg.norm(W1, axis=0)
    top_indices = np.argsort(norms)[::-1][:n_show]

    ncols = int(np.ceil(np.sqrt(n_show)))
    nrows = int(np.ceil(n_show / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if idx < n_show:
            neuron_idx = top_indices[idx]
            w = W1[:, neuron_idx].reshape(image_size, image_size, 3)
            w_min, w_max = w.min(), w.max()
            if w_max - w_min > 1e-8:
                w_norm = (w - w_min) / (w_max - w_min)
            else:
                w_norm = np.zeros_like(w)
            ax.imshow(w_norm)
            ax.set_title(f'N{neuron_idx}', fontsize=8)
        ax.axis('off')

    plt.suptitle('First Layer Weight Visualization (Top Neurons by Norm)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Weight visualization saved to {save_dir}/weight_visualization.png")


def visualize_errors(misclassified_path, data_dir, save_dir='figures', n_show=16):
    os.makedirs(save_dir, exist_ok=True)

    data = np.load(misclassified_path, allow_pickle=True)
    true_labels = data['true_labels']
    pred_labels = data['pred_labels']
    paths = data['paths']

    n_show = min(n_show, len(true_labels))
    if n_show == 0:
        print("No misclassified samples to visualize.")
        return

    indices = np.random.choice(len(true_labels), n_show, replace=False)

    ncols = int(np.ceil(np.sqrt(n_show)))
    nrows = int(np.ceil(n_show / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    from PIL import Image

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if idx < n_show:
            sample_idx = indices[idx]
            img_path = str(paths[sample_idx])
            true_name = CLASS_NAMES[true_labels[sample_idx]]
            pred_name = CLASS_NAMES[pred_labels[sample_idx]]

            img = Image.open(img_path).convert('RGB')
            ax.imshow(np.array(img))
            ax.set_title(f'T:{true_name[:8]}\nP:{pred_name[:8]}', fontsize=8, color='red')
        ax.axis('off')

    plt.suptitle('Misclassified Samples (True vs Predicted)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error analysis saved to {save_dir}/error_analysis.png")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--history_path', type=str, default='checkpoints/history.json')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.npz')
    parser.add_argument('--cm_path', type=str, default='results/confusion_matrix.npy')
    parser.add_argument('--misclassified_path', type=str, default='results/misclassified.npz')
    parser.add_argument('--data_dir', type=str, default='../EuroSAT_RGB')
    parser.add_argument('--save_dir', type=str, default='figures')
    args = parser.parse_args()

    if os.path.exists(args.history_path):
        plot_training_curves(args.history_path, args.save_dir)

    if os.path.exists(args.model_path):
        visualize_first_layer_weights(args.model_path, save_dir=args.save_dir)

    if os.path.exists(args.cm_path):
        plot_confusion_matrix(args.cm_path, args.save_dir)

    if os.path.exists(args.misclassified_path):
        visualize_errors(args.misclassified_path, args.data_dir, args.save_dir)