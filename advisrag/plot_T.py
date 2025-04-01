import numpy as np

t_list = [10 * i for i in range(1, 6)]
loaded_scores2 = np.load('scores2.npy')
scaled_scores = [loaded_scores2 * t for t in t_list]
alpha = [np.exp(s) / np.sum(np.exp(s), axis=1, keepdims=True) for s in scaled_scores]

# unscaled_alpha2 = torch.softmax(scores2, dim=1)
unscaled_alpha2 = np.exp(loaded_scores2) / np.sum(np.exp(loaded_scores2), axis=1, keepdims=True)

# Plot the distributions
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 18))

indices = range(1, len(unscaled_alpha2[0]) + 1, len(unscaled_alpha2[0]) // 4)

# Plot unscaled_alpha2[10]
plt.subplot(3, 2, 1)
plt.bar(range(1, len(unscaled_alpha2[0]) + 1), unscaled_alpha2[10], alpha=0.5, label='unscaled_alpha2')
plt.title('Original')
plt.xlabel('Sub-Image')
plt.ylabel('Attention Score')
plt.xticks(indices)  # Set x-axis to show integer ticks
# plt.legend()

# Plot each element in alpha list
for i in range(5):
    plt.subplot(3, 2, i + 2)
    plt.bar(range(1, len(alpha[i][10]) + 1), alpha[i][10], alpha=0.5, label=f'alpha[{i}]')
    plt.title(f'T = {t_list[i]}')
    plt.xlabel('Sub-Image')
    plt.ylabel('Attention Score')
    plt.xticks(indices)  # Set x-axis to show integer ticks
    # plt.legend()

plt.tight_layout(h_pad=5.0, w_pad=2.0)
plt.show()