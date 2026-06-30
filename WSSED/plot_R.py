from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
results = pkl.load(open('temps/data/ablation_results.pkl', 'rb'))
hipool_result = pkl.load(open('temps/hi_pool_result.pkl', 'rb'))
fig, ax_events = plt.subplots(2, 3, figsize=(14, 4), dpi=200, sharex=True, sharey=True)
n_events = 17
col = list(range(2))
row = list(range(3))
labels = [
      "Train horn", "Air horn, truck horn", "Car alarm", "Reversing beeps", "Bicycle", "Skateboard",
      "Ambulance (siren)", "Fire engine, fire truck (siren)", "Civil defense siren", "Police car (siren)",
      "Screaming", "Car", "Car passing by", "Bus", "Truck", "Motorcycle", "Train"
    ]
IDX = [0, 11, 16]
k = 0
Rid = [1, 12, 8]
# 0	 1	4	4	7	6	0	0	5	0	5	8	8	7	7	7	6
R = ['max', 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 40, 48, 60, 80, 120, 'avg', '', 'R*']
for r in row:
    tag_data = []
    loc_data = []
    i = IDX[r]
    for n in range(20):
        tag_data.append(results[n][0][0][i])
        loc_data.append(list(results[n][1].values())[i]['f_measure']['f_measure'])
    for n in range(len(tag_data)):
        ax_events[0][r].bar(n, tag_data[n], label='F1 on Tagging', color=['mediumseagreen'])
        ax_events[1][r].bar(n, loc_data[n], label='F1 on Localization', color=['royalblue'])
    
    ax_events[0][r].bar(21, hipool_result[0][0][i], color=['gold'])
    ax_events[1][r].bar(21, list(hipool_result[1].values())[i]['f_measure']['f_measure'], color=['gold'])

    ax_events[0][r].plot([hipool_result[0][0][i]] * 22, c='gold', linestyle='--', lw=2)
    ax_events[1][r].plot([list(hipool_result[1].values())[i]['f_measure']['f_measure']] * 22, c='gold', linestyle='--', lw=2)
    ax_events[0][r].set_title(labels[i], fontdict={'family': 'Times New Roman', 'size':13})
    ax_events[1][r].set_title(labels[i], fontdict={'family': 'Times New Roman', 'size':13})
    ax_events[0][0].set_ylim([0. , 1.])
    ax_events[1][0].set_ylim([0. , 1.])
    ax_events[0][r].set_xticks(list(np.arange(0, 1.2, 0.25)))
    ax_events[1][r].set_xticks(list(np.arange(0, 1.2, 0.25)))

    ax_events[0][r].set_yticklabels(list(np.arange(0., 1.2, 0.25)), fontdict={'family': 'Times New Roman', 'size':11})
    ax_events[1][r].set_yticklabels(list(np.arange(0., 1.2, 0.25)), fontdict={'family': 'Times New Roman', 'size':11})
    ax_events[1][r].set_xticks(list(range(0, 22)))
    ax_events[1][r].set_xticklabels(['max', 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 40, 48, 60, 80, 120, 'avg', '', 'R*']
                                    , rotation=35, fontdict={'family': 'Times New Roman', 'size':11})
    ax_events[1][r].set_xlabel('The number of sub-bags R in HiPool', fontdict={'family': 'Times New Roman', 'size':13})
# ax_events[-1][-1].axis('off')
# ax_events[-1][-1].axis('off')
handles0, labels0 = ax_events[0][0].get_legend_handles_labels()
handles1, labels1 = ax_events[1][0].get_legend_handles_labels()

ax_events[0][0].legend(handles0[0], labels0, loc='upper left', ncol=1, fancybox=True,
                       prop={'family': 'Times New Roman', 'size':11})
ax_events[1][0].legend(handles1[0], labels1, loc='upper left', ncol=1, fancybox=True,
                       prop={'family': 'Times New Roman', 'size':11})

ax_events[0][0].set_ylabel('F1 score', fontdict={'family': 'Times New Roman', 'size':13})
ax_events[1][0].set_ylabel('F1 score', fontdict={'family': 'Times New Roman', 'size':13})

fig.tight_layout()
plt.savefig('result.png')

