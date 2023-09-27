import argparse
import numpy as np
import json
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualization')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('-f', '--file', help='json file path')

    args = parser.parse_args()

    seed = args.seed
    json_file = args.file

    with open(json_file, 'r') as f:
        data = json.load(f)
    r_pre = []
    r_pos = []
    r_neg = []
    for d in data.keys():
        r_pre.append(data[d]['premise'])
        r_pos += data[d]['positives']
        r_neg += data[d]['negatives']

    x = np.array(r_pre+r_pos+r_neg)
    split = [0, len(r_pre), len(r_pre+r_pos), len(r_pre+r_pos+r_neg)]
    labels = ['pre', 'pos', 'neg']

    color = list(plt.cm.tab10.colors)

    tsne = TSNE(random_state=seed)
    y = tsne.fit_transform(x)
    plt.figure(figsize=(15, 9))
    plt.rcParams['font.family'] = 'Times New Roman'
    for i in range(0, len(split) - 1):
        plt.scatter(y[split[i]:split[i+1], 0], y[split[i]:split[i+1], 0], label=labels[i], color=color[i])
    plt.axis('off')
    #plt.gca().invert_xaxis()
    plt.legend(fontsize=20, markerscale=2.0, loc='upper right')
    plt.tight_layout(pad=0.0)
    plt.savefig('tsne.pdf')
    plt.close()

    umap = UMAP(random_state=seed)
    y = umap.fit_transform(x)
    plt.figure(figsize=(15, 9))
    plt.rcParams['font.family'] = 'Times New Roman'
    for i in range(0, len(split) - 1):
        plt.scatter(y[split[i]:split[i + 1], 0], y[split[i]:split[i + 1], 0], label=labels[i], color=color[i])
    plt.axis('off')
    # plt.gca().invert_xaxis()
    plt.legend(fontsize=20, markerscale=2.0, loc='upper right')
    plt.tight_layout(pad=0.0)
    plt.savefig('umap.pdf')
    plt.close()
