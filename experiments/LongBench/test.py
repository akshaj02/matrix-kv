import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load logs
with open("cake_debug_scores.pkl", "rb") as f:
    logs = pickle.load(f)

# Choose a layer and step
layer = 0
step = 1
print("Available layers:", sorted(logs.keys()))

# Get attention matrix: [batch, heads, query_len, key_len]
attn = logs[layer]['attn_heatmap'][step]
num_heads = attn.shape[1]

# Create subplot grid dynamically based on head count
ncols = 4
nrows = (num_heads + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
axes = axes.flatten()
num_heads = min(num_heads, len(axes))  # Limit to available axes

for head in range(num_heads):
    attn_matrix = attn[0, head]
    sns.heatmap(
    attn_matrix.numpy(),
    cmap="magma",
    ax=axes[head],
    cbar=False,
    vmin=0.0,
    vmax=0.01  # saturate anything above 1%
    )
    axes[head].set_title(f"Head {head}")
    axes[head].set_xlabel("Key")
    axes[head].set_ylabel("Query")

plt.suptitle(f"Layer {layer} | All {num_heads} Attention Heads | Step {step}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"attn_heatmap_layer{layer}_all_heads_step{step}.png", dpi=300)
plt.close()

print(f"\n[✓] Saved: attn_heatmap_layer{layer}_all_heads_step{step}.png")
