from pyAudioAnalysis import audioFeatureExtraction

# F = audioFeatureExtraction.dirWavFeatureExtraction('/home/lib/audio/work/fm_2018_collected_singles/monoblock/wav/', 1.0, 0.5, 0.05, 0.025)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sklearn.manifold as skmf

algos = [
    ['Isomap', skmf.Isomap],
    ['LocallyLinearEmbedding', skmf.LocallyLinearEmbedding],
    ['MDS', skmf.MDS],
    ['SpectralEmbedding', skmf.SpectralEmbedding],
    ['TSNE', skmf.TSNE],
    ['PCA', skmf.t_sne.PCA],
]

X = F[0][:,[2,5,6]]

fig = plt.figure()
gs = GridSpec(2, len(algos)//2)

for i, algo in enumerate(algos):
    t_tmp = algo[1]()
    f_tmp = t_tmp.fit_transform(X)
    algo.append(f_tmp.copy())

    ax = fig.add_subplot(gs[i])
    ax.set_title(algo[0])
    # ax.plot(f_tmp[:,0], f_tmp[:,1], 'o', alpha=0.5, label=algo[0])
    # ax.legend()

    g_tmp = nx.MultiDiGraph()
    for n in [_.split(' --- ')[-1] for _ in F[1]]:
        g_tmp.add_node(n)
    pos = nx.drawing.layout.random_layout(g_tmp)

    for i, n in enumerate([_.split(' --- ')[-1] for _ in F[1]]):
        pos[n] = f_tmp[i,:2]

    nx.draw_networkx_nodes(g_tmp, pos, alpha=0.5)
    nx.draw_networkx_labels(g_tmp, pos, font_size=8)
    


idx_spectral_entropy = np.argsort(F[0][:,5])

fig = plt.figure()
for i, idx in enumerate(idx_spectral_entropy):
    plt.plot(F[0][idx,[5]], [i%4*0.002-0.003], 'o')
    plt.text(F[0][idx,5], i%4*0.002-0.003, F[1][idx].split(' --- ')[-1][:-4])
plt.xlim((0.06210913036933147, 1.908263048423288))
plt.xlabel(F[2][5])

