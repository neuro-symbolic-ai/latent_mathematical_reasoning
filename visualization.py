import argparse
import numpy as np
import json
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualization')
    parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')
    parser.add_argument('-f', '--file', default='test_set_cross_embeddings.json', help='json file path')

    args = parser.parse_args()

    seed = args.seed
    json_file = args.file

    with open(json_file, 'r') as f:
        data = json.load(f)

    r_pre = []
    r_pos = []
    r_neg = []
    split = {}
    i = 0
    for d in data.keys():
        r_pre.append(data[d]['premise'])
        r_pos += data[d]['positives']
        r_neg += data[d]['negatives']
        split[d] = [(len(r_pre) - 1, len(r_pre)),
                    (len(r_pos) - len(data[d]['positives']), len(r_pos)),
                    (len(r_neg) - len(data[d]['negatives']), len(r_neg))]
        i += 1
        if i == 20:
            break

    x = np.array(r_pre+r_pos+r_neg)

    tsne = TSNE(perplexity=50, metric = "cosine", early_exaggeration = 20.0, random_state=seed)
    y_tsne = tsne.fit_transform(x)

    umap = UMAP(n_neighbors=500, random_state=seed)
    y_umap = umap.fit_transform(x)

    for d in ['2']:
        temp = split[d]
        temp = [(temp[0][0], temp[0][1]),
                (temp[1][0] + len(r_pre), temp[1][1] + len(r_pre)),
                (temp[2][0] + len(r_pre) + len(r_pos), temp[2][1] + len(r_pre) + len(r_pos))]
        labels = ['pre', 'pos', 'neg']
        markers = ['o', '+', 'x']

        color = list(plt.cm.tab10.colors)

        plt.figure(figsize=(4, 4))
        for i in range(0, len(labels)):
            plt.scatter(y_tsne[temp[i][0]:temp[i][1], 0], y_tsne[temp[i][0]:temp[i][1], 1],
                        marker=markers[i], label=labels[i], color=color[i])
        plt.legend(fontsize=10, markerscale=1.0)
        plt.savefig(f'{json_file[:-5]}_tsne_{d}.png')
        plt.close()
