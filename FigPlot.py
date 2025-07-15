import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np

sign = ['(a)', '(b)', '(c)', '(d)', '(e)']

data_total = []
data_name = ['model', 'dijkstra', 'bit', 'rrt']

for i in range(len(data_name)):
    data_path = 'data/data_' + data_name[i] + '.npy'
    data_total.append(np.load(data_path))

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=1, wspace=0.3)
axes = fig.subplots(nrows=1, ncols=5)

sampling = [200, 400, 600, 800, 1000]

label = ['GNN model', 'PRM', 'BIT*', 'RRT*']
color = ['green', 'blue', 'red', 'pink']
marker = ['o', 'X', 'p', 'D']
tit = ['Collision Check', 'Total Time(seconds)', 'Planning Time(seconds)', 'Path Cost', 'Success Rate']

for i in range(len(data_total)):
    data_total[i][:, -1] = data_total[i][:, -1] / 100

for i in range(len(data_total)):
    axes[0].errorbar(sampling, data_total[i][:, 2 * 0], yerr=data_total[i][:, 1], color=color[i], marker=marker[i],
                     label=label[i])
    axes[-1].errorbar(sampling, data_total[i][:, -1], yerr=0, color=color[i], marker=marker[i])

    for j in range(1, len(data_total[0])):
        if j < len(data_total[0]) - 1:
            axes[j].errorbar(sampling, data_total[i][:, 2 * j], yerr=data_total[i][:, 2 * j + 1], color=color[i],
                             marker=marker[i])

for i in range(5):
    axes[i].set_title(tit[i])
    axes[i].set_xlabel('sampling number')
    axes[i].set_xlim([100, 1100])
    axes[i].set_xticks([200, 400, 600, 800, 1000])
    axes[i].grid()
    # axes[i].set_title(sign[i], y=-0.4)

axes[-1].set_ylim([0.8, 1])
axes[-1].set_yticks(np.linspace(0.8, 1, 5))

lines = []
labels = []
for ax in fig.axes:
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)

fig.legend(lines, labels, ncol=len(data_total), loc='upper center')
plt.subplot_tool()
plt.show()

print(data_total)

# ---------------------------- test 2 ------------------------------

# fig = plt.figure(figsize=(15, 3))
# fig.subplots_adjust(hspace=1, wspace=0.3)
# axes = fig.subplots(nrows=1, ncols=4)
#
# data_model_6 = np.hstack([np.load('data/data_model_6.npy')[0], [0]])
# data_bit_6 = np.hstack([np.load('data/data_bit_6.npy')[0], [0]])
# data_rrt_6 = np.hstack([np.load('data/data_rrt_6.npy')[0], [0]])
# data_prm_6 = np.hstack([np.load('data/data_prm_6.npy')[0], [0]])
#
#
# size = 6
# x = np.arange(size)
# width = 0.3
#
# tit = ['Collision Check', 'Planning Time', 'Path Cost', 'Success Rate']
#
# for i in range(0, 4):
#
#     axes[i].errorbar(1, data_model_6[2 * (i + 1)], color=color[0], capsize=6,
#                      linestyle="None",
#                      marker="s", markersize=5, mfc=color[0], mec=color[0], yerr=data_model_6[2 * (i + 1) + 1] + [0])
#     axes[i].errorbar(2, data_prm_6[2 * (i + 1)], color=color[1], capsize=6,
#                      linestyle="None",
#                      marker="s", markersize=5, mfc=color[1], mec=color[1], yerr=data_prm_6[2 * (i + 1) + 1] + [0])
#     axes[i].errorbar(3, data_bit_6[2 * (i + 1)], color=color[2], capsize=6,
#                      linestyle="None",
#                      marker="s", markersize=5, mfc=color[2], mec=color[2], yerr=data_bit_6[2 * (i + 1) + 1] + [0])
#     axes[i].errorbar(4, data_rrt_6[2 * (i + 1)], color=color[3], capsize=6,
#                      linestyle="None",
#                      marker="s", markersize=5, mfc=color[3], mec=color[3], yerr=data_rrt_6[2 * (i + 1) + 1] + [0])
#
# for i in range(4):
#     axes[i].set_xticks(x)
#     axes[i].set_title(tit[i])
#     axes[i].set_xticklabels(['', 'Model', 'PRM', 'BIT*', 'RRT*', ''])
#     axes[i].grid()
#     axes[i].set_title(sign[i], y=-0.25)
#
# axes[-1].set_yticks(np.arange(80, 101, 5))
# axes[-2].set_yticks(np.arange(0, 21, 5))
#
# plt.savefig('7.png')
# plt.subplot_tool()
# plt.show()
#
#
# print(data_model_6)
# print(data_prm_6)
# print(data_bit_6)
