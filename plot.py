import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.weight"] = 400  # 可选：400 为正常，700 为粗体


# 数据
datasets = ["PlotQA", "InfoVQA", "DocVQA"]
method = ["top-10", "top-3", "MAC"]

# Runtime 数据
runtime = {
    "PlotQA": [12.54, 3.99, 7.04],
    "InfoVQA": [3.56, 1.12, 1.92],
    "DocVQA": [3.87, 1.22, 2.05],
}

# Accuracy 数据
accuracy = {
    "PlotQA": [32.96, 40.92, 44.57],
    "InfoVQA": [37.73, 52.39, 58.07],
    "DocVQA": [60.56, 76.53, 84.59],
}

bar_width = 0.25
x = np.arange(len(datasets))
offsets = [-bar_width, 0, bar_width]
# 使用更为专业的颜色方案
# bar_colors = ["#1f77b4", "#2ca02c", "#d62728"]
bar_colors = ["#1f77b4", "#8bc34a", "#d62728"]

# 创建一个带有双 y 轴的图表
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

# 绘制条形图（运行时）并添加标注
for i in range(len(method)):
    run_vals = [runtime[ds][i] for ds in datasets]
    rects = ax1.bar(
        x + offsets[i],
        run_vals,
        width=bar_width,
        color=bar_colors[i],
        edgecolor="#070d0d",
        linewidth=1,
        label=f"{method[i]}",
    )

    # 添加标注
    for rect in rects:
        height = rect.get_height()
        ax1.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#070d0d",
        )

# 绘制折线图（准确率）并添加标注
for j, ds in enumerate(datasets):
    x_coords = x[j] + np.array(offsets)
    acc_vals = accuracy[ds]

    ax2.plot(
        x_coords,
        acc_vals,
        marker=r"D",
        markersize=6,
        linestyle="-",
        color="orange",
        markeredgecolor="#070d0d",
        markerfacecolor="darkorange",
        linewidth=1.5,
        label=f"{ds} Accuracy",
    )

    # 添加标注
    for k, (x_coord, acc) in enumerate(zip(x_coords, acc_vals)):
        ax2.text(
            x_coord,
            (
                acc + 2.2 if ds == "PlotQA" else acc - 2.5
            ),  # 调整 PlotQA 的标注位置避免重叠
            f"{acc:.1f}",
            ha="center",
            va="bottom" if ds != "PlotQA" else "top",
            fontsize=9,
            color="#070d0d",
        )


# 设置坐标轴标签和刻度
ax1.set_xlabel("Datasets", fontsize=12, fontweight='bold')
ax1.set_ylabel("Runtime (h)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=10)
plt.title("Runtime and Accuracy Comparison", fontsize=14, pad=20, fontweight="bold")

# 添加图例
ax1.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.06),
    ncol=3,
    frameon=True,
    borderpad=0.18,
    framealpha=1,
)

# 确保布局紧凑
plt.tight_layout()

# 显示网格线
ax1.grid(axis="y", linestyle="--", alpha=0.7, linewidth=0.5)
ax2.grid(False)  # 隐藏右边 y 轴的网格线

plt.savefig('runtime_accuracy_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 载入数据
t_list = [10 * i for i in range(1, 6)]
loaded_scores2 = np.load("scores2.npy")
scaled_scores = [loaded_scores2 * t for t in t_list]
alpha = [np.exp(s) / np.sum(np.exp(s), axis=1, keepdims=True) for s in scaled_scores]

# 计算未缩放的 alpha
unscaled_alpha2 = np.exp(loaded_scores2) / np.sum(
    np.exp(loaded_scores2), axis=1, keepdims=True
)

# 设置字体和图表样式
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体
plt.rcParams['font.size'] = 10        # 设置字体大小

# 创建一个大型的图形区域
plt.figure(figsize=(8, 9))  # 调整图表尺寸

# 确定子图位置和标签
indices = range(1, len(unscaled_alpha2[0]) + 1, len(unscaled_alpha2[0]) // 4)

# 绘制原始分布
plt.subplot(3, 2, 1)
bar1 = plt.bar(
    range(1, len(unscaled_alpha2[0]) + 1),
    unscaled_alpha2[10],
    color='#1f77b4',  # 使用更专业的颜色
    alpha=1,
    label="Original"
)
plt.title("Original Attention Distribution", fontsize=12, fontweight='bold')
plt.xlabel("Sub-Image Index", fontsize=10)
plt.ylabel("Attention Score", fontsize=10)
plt.xticks(indices)
plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线



import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
# cmap = plt.get_cmap('Reds')  # 使用'Reds' colormap
cmap = LinearSegmentedColormap.from_list("custom_red", ['#ff8787', '#d62728'])
max_temp_index = 4  # 最大的温度对应最深的红色


# 绘制不同温度下的分布
for i in range(5):
    plt.subplot(3, 2, i + 2)
    color_value = cmap(i / max_temp_index)
    bar = plt.bar(
        range(1, len(alpha[i][10]) + 1),
        alpha[i][10],
        color=color_value,  # 使用对比色
        alpha=1,
        label=f"T = {t_list[i]}"
    )
    # plt.title(f"Temperature T = 1 / {t_list[i]}", fontsize=12, fontweight='bold')
    plt.title(rf"Temperature $\tau$ = 1 / {t_list[i]}", fontsize=12, fontweight='bold')
    plt.xlabel("Sub-Image Index", fontsize=10)
    plt.ylabel("Attention Score", fontsize=10)
    plt.xticks(indices)
    plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线

# 调整布局
plt.tight_layout(h_pad=1.5, w_pad=2.0)

# 保存图表
plt.savefig("attention_distribution.pdf", dpi=300, bbox_inches='tight')

plt.show()
