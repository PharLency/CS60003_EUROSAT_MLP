import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from data_loader import CLASS_NAMES

save_dir = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(os.path.dirname(__file__), 'results_aug', 'test_results.json')) as f:
    res = json.load(f)

classes = list(res['per_class_acc'].keys())
accs = [res['per_class_acc'][c] for c in classes]
short = [c[:10] for c in classes]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(classes)), [a * 100 for a in accs], color='steelblue', edgecolor='black')
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(short, rotation=35, ha='right', fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Per-Class Test Accuracy', fontsize=14)
ax.set_ylim(0, 100)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
            f'{acc * 100:.1f}%', ha='center', fontsize=9)
ax.axhline(y=res['test_acc'] * 100, color='red', linestyle='--', linewidth=1.5,
           label=f'Overall: {res["test_acc"] * 100:.2f}%')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), dpi=150)
plt.close()
print("per_class_accuracy.png saved.")

with open(os.path.join(os.path.dirname(__file__), 'search_results_aug', 'search_results.json')) as f:
    gs = json.load(f)

lrs = sorted(set(r['lr'] for r in gs['results']))
hiddens = sorted(set(r['hidden1_dim'] for r in gs['results']))

heatmap = np.zeros((len(lrs), len(hiddens)))
for r in gs['results']:
    li = lrs.index(r['lr'])
    hi = hiddens.index(r['hidden1_dim'])
    heatmap[li, hi] = max(heatmap[li, hi], r['best_val_acc'])

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heatmap * 100, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(hiddens)))
ax.set_xticklabels([str(h) for h in hiddens])
ax.set_yticks(range(len(lrs)))
ax.set_yticklabels([str(l) for l in lrs])
ax.set_xlabel('Hidden Dimension', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('Grid Search: Best Val Accuracy (%)', fontsize=14)
for i in range(len(lrs)):
    for j in range(len(hiddens)):
        ax.text(j, i, f'{heatmap[i, j] * 100:.1f}', ha='center', va='center', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, label='Val Acc (%)')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'grid_search_heatmap.png'), dpi=150)
plt.close()
print("grid_search_heatmap.png saved.")

with open(os.path.join(os.path.dirname(__file__), 'search_results_aug', 'random_search_results.json')) as f:
    rs = json.load(f)

results = rs['results']
log_lrs = np.array([np.log10(r['lr']) for r in results])
val_accs = np.array([r['best_val_acc'] * 100 for r in results])
hidden_sizes = np.array([r['hidden1_dim'] for r in results])
activations = [r['activation'] for r in results]

fig, ax = plt.subplots(figsize=(9, 6))
color_map = {'relu': 'tab:blue', 'tanh': 'tab:orange'}
size_map = 40 + hidden_sizes / 4.0

for act in ['relu', 'tanh']:
    idx = [i for i, a in enumerate(activations) if a == act]
    if not idx:
        continue
    ax.scatter(
        log_lrs[idx],
        val_accs[idx],
        s=size_map[idx],
        c=color_map[act],
        alpha=0.75,
        edgecolors='black',
        linewidths=0.6,
        label=act.upper(),
    )

best_idx = int(np.argmax(val_accs))
best_r = results[best_idx]
ax.scatter(
    [log_lrs[best_idx]],
    [val_accs[best_idx]],
    s=size_map[best_idx] + 60,
    facecolors='none',
    edgecolors='crimson',
    linewidths=2.0,
)
ax.annotate(
    (
        f"Best: {val_accs[best_idx]:.2f}%\n"
        f"lr={best_r['lr']:.4f}, h1={best_r['hidden1_dim']}, h2={best_r['hidden2_dim']}"
    ),
    (log_lrs[best_idx], val_accs[best_idx]),
    xytext=(10, 10),
    textcoords='offset points',
    fontsize=9,
    color='crimson',
)

ax.set_xlabel('log10(Learning Rate)', fontsize=12)
ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
ax.set_title('Random Search Results', fontsize=14)
ax.grid(True, alpha=0.3)
act_legend = ax.legend(fontsize=11, loc='upper left')
ax.add_artist(act_legend)

size_handles = []
for hidden in [128, 512, 1024]:
    size_handles.append(
        plt.scatter([], [], s=40 + hidden / 4.0, c='gray', alpha=0.5, edgecolors='black', linewidths=0.6)
    )
size_labels = ['h1=128', 'h1=512', 'h1=1024']
legend2 = ax.legend(size_handles, size_labels, title='Hidden1 size', loc='lower right', fontsize=10)
ax.add_artist(legend2)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'random_search_scatter.png'), dpi=150)
plt.close()
print("random_search_scatter.png saved.")

counts = [3000, 3000, 3000, 2500, 2500, 2000, 2500, 3000, 2500, 3000]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(CLASS_NAMES)), counts, color='teal', edgecolor='black')
ax.set_xticks(range(len(CLASS_NAMES)))
ax.set_xticklabels([c[:10] for c in CLASS_NAMES], rotation=35, ha='right', fontsize=10)
ax.set_ylabel('Number of Images', fontsize=12)
ax.set_title('EuroSAT Dataset Class Distribution', fontsize=14)
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 30, str(c), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'dataset_distribution.png'), dpi=150)
plt.close()
print("dataset_distribution.png saved.")

print("All extra figures generated.")
