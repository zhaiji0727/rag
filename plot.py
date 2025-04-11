import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 600  # 可选：400 为正常，700 为粗体
plt.rcParams['figure.dpi'] = 300


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
bar_colors = ["#1f77b4", "#2ca02c", "#d62728"]

# 创建一个带有双 y 轴的图表
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

'''
# 绘制运行时条形图在左边 y 轴上，增加描边
for i in range(len(method)):
    run_vals = [runtime[ds][i] for ds in datasets]
    # 添加黑色边框
    ax1.bar(
        x + offsets[i],
        run_vals,
        width=bar_width,
        color=bar_colors[i],
        edgecolor='#070d0d',  # 
        linewidth=1,       # 设置描边宽度
        label=f"{method[i]}",
    )

# 为每个数据集绘制 accuracy 的线图
for j, ds in enumerate(datasets):
    x_coords = x[j] + np.array(offsets)
    acc_vals = accuracy[ds]
    # 使用更为清晰的线条和标记样式
    ax2.plot(x_coords, acc_vals,
            marker=r"D",
            markersize=6,
            linestyle="-",
            color="orange",          # 线条颜色
            markeredgecolor="#070d0d",  # 标记边缘（描边）颜色
            markerfacecolor="darkorange",  # 标记内部填充颜色
            linewidth=1.5)
'''
# 绘制条形图（运行时）并添加标注
for i in range(len(method)):
    run_vals = [runtime[ds][i] for ds in datasets]
    rects = ax1.bar(
        x + offsets[i],
        run_vals,
        width=bar_width,
        color=bar_colors[i],
        edgecolor='#070d0d',
        linewidth=1,
        label=f"{method[i]}",
    )

    # 添加标注
    for rect in rects:
        height = rect.get_height()
        ax1.text(
            rect.get_x() + rect.get_width()/2.,
            height,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=9,
            color='#070d0d'
        )

# 绘制折线图（准确率）并添加标注
for j, ds in enumerate(datasets):
    x_coords = x[j] + np.array(offsets)
    acc_vals = accuracy[ds]

    ax2.plot(x_coords, acc_vals,
             marker=r"D",
             markersize=6,
             linestyle="-",
             color="orange",
             markeredgecolor="#070d0d",
             markerfacecolor="darkorange",
             linewidth=1.5,
             label=f"{ds} Accuracy")

    # 添加标注
    for k, (x_coord, acc) in enumerate(zip(x_coords, acc_vals)):
        ax2.text(
            x_coord,
            acc + 2 if ds == "PlotQA" else acc - 2.3,  # 调整 PlotQA 的标注位置避免重叠
            f'{acc:.1f}',
            ha='center',
            va='bottom' if ds != "PlotQA" else 'top',
            fontsize=9,
            color='#070d0d'
        )


# 设置坐标轴标签和刻度
ax1.set_xlabel("Datasets", fontsize=12)
ax1.set_ylabel("Runtime (h)", fontsize=12)
ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=10)
plt.title("Runtime and Accuracy Comparison", fontsize=14, pad=20)

# 添加图例
ax1.legend(loc="upper center",
          bbox_to_anchor=(0.5, 1.15),
          ncol=3,
          frameon=False)

# 确保布局紧凑
plt.tight_layout()

# 显示网格线
ax1.grid(axis='y',
         linestyle='--',
         alpha=0.7,
         linewidth=0.5)
ax2.grid(False)  # 隐藏右边 y 轴的网格线

plt.show()
