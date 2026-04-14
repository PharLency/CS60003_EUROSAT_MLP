import numpy as np
import os
import sys

from train import train
from test import evaluate
from visualize import plot_training_curves, visualize_first_layer_weights, plot_confusion_matrix, visualize_errors


def main():
    data_dir = os.environ.get('DATA_DIR', '../EuroSAT_RGB')
    save_dir = os.environ.get('SAVE_DIR', 'checkpoints')
    results_dir = os.environ.get('RESULTS_DIR', 'results')
    figures_dir = os.environ.get('FIGURES_DIR', 'figures')
    seed = int(os.environ.get('SEED', '42'))

    print("=" * 60)
    print("Step 1: Training")
    print("=" * 60)
    model, history, data = train(
        data_dir=data_dir,
        save_dir=save_dir,
        hidden1_dim=512,
        hidden2_dim=256,
        activation='relu',
        lr=0.005,
        weight_decay=1e-4,
        batch_size=128,
        epochs=50,
        lr_step_size=15,
        lr_gamma=0.5,
        seed=seed,
        augment=True,
    )

    print("\n" + "=" * 60)
    print("Step 2: Testing")
    print("=" * 60)
    results, cm, misclassified = evaluate(
        data_dir=data_dir,
        model_path=os.path.join(save_dir, 'best_model.npz'),
        save_dir=results_dir,
        seed=seed,
    )

    print("\n" + "=" * 60)
    print("Step 3: Visualization")
    print("=" * 60)
    plot_training_curves(os.path.join(save_dir, 'history.json'), figures_dir)
    visualize_first_layer_weights(os.path.join(save_dir, 'best_model.npz'), save_dir=figures_dir)
    plot_confusion_matrix(cm, figures_dir)
    visualize_errors(os.path.join(results_dir, 'misclassified.npz'), data_dir, figures_dir)

    print("\n" + "=" * 60)
    print("All done!")
    print(f"  Checkpoints: {save_dir}/")
    print(f"  Results:     {results_dir}/")
    print(f"  Figures:     {figures_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()