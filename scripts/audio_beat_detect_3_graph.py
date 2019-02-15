"""detect onsets, beats, segments in audio

Graphical version of music_beat_detect_2.py, starting from librosa
"onset times from a signal" example, extended a bit with additional
variations and algorithms.

Using librosa, madmom, essentia

## TODO
- persistent engine
- iterative / interactive run
- incremental graph expansion
- bayesian ensembles for event constraints and integration
"""

import argparse, time, sys, os, pprint
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import madmom

import networkx as nx

from scipy.stats import mode
from smp_base.plot import make_fig

DEBUG=True

from slurp.common import myprint
from slurp.common import data_load_librosa, compute_chroma_librosa

import ge.engine

# from PyQt5.
# from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5

def getgraph():
    """get graph (engine) reference

    - connect to ipython kernel
     - get existing graph
     - get existing node (by signature) OR
     - create and run existing node
    - create and run graph locally
    """
    pass

def main_graph(args):
    """main beat detection graph version

    - basic graph repr: networkx, floppy
    - basic graph construction loop:
      - test condition
      - create node
      - compute node and create signals

    1. define-and-run
    2. define-by-run
    """
    graph_conf = {
        'name': 'g',
        'nodes': {
            'input-file': {},
            'input-duration': {},
            'data-loader': {},
            'comp-chroma': {},
        },
        'edges': {
        },
    }
    e = ge.engine.engine()
    e.load(graph_conf)
    e.run()

    print('\n')
    
    g = nx.MultiDiGraph(name=sys.argv[0].split('/')[-1][:-3])

    g.add_node(g.number_of_nodes(), name='input-file', file=args.file)
    g.add_node(g.number_of_nodes(), name='input-duration', duration=args.duration)
    
    g.add_node(g.number_of_nodes(), name='data-loader', func=data_load_librosa, outputs=['y', 'sr'])
    g.add_edges_from([(0, 2), (1, 2)])

    y, sr = g.node[2]['func'](filename=args.file, duration=75)
    print(('y = {0}, sr = {1}'.format(y, sr)))
    
    g.add_node(g.number_of_nodes(), name='comp-chroma', func=compute_chroma_librosa, outputs=['C'])
    g.add_edges_from([(2, 3)])
    C = g.node[3]['func'](y=y, sr=sr)
    print(('C = {0}'.format(C)))
    
    print(('{1:>20} = {0}'.format(g, 'g')))
    print(('{1:>20} = {0}'.format(pprint.pformat(dict(g.nodes.data())), 'g.nodes.data')))
    print(('{1:>20} = {0}'.format(pprint.pformat(nx.to_dict_of_dicts(g)), 'g.to_dict_of_dicts')))
    print(('{1:>20} = {0}'.format(pprint.pformat(nx.to_dict_of_lists(g)), 'g.to_dict_of_lists')))
    print(('{1:>20} = {0}'.format(pprint.pformat(dict(g.node(data=True))), 'g.node(data=True)')))

    # g2 = nx.MultiDiGraph(g.node(data=True))
    # print(g2)
    # print('{1:>20} = {0}'.format(pprint.pformat(dict(g2.node(data=True))), 'g2.node(data=True)'))
    
    g_pos = nx.layout.random_layout(g)
    g_labels = dict([(gk, gv['name']) for gk, gv in g.nodes.data()])

    # qApp = QtWidgets.QApplication([b"matplotlib"])

    plt.ion()
    fig = plt.figure()
    fig.show()
    plt.show()
    
    ax = fig.add_subplot(1,1,1)
    nx.draw_networkx_nodes(g, g_pos, ax=ax, alpha=0.5)
    nx.draw_networkx_labels(g, g_pos, ax=ax, alpha=0.8, labels=g_labels)
    nx.draw_networkx_edges(g, g_pos, ax=ax, alpha=0.5)

    ax.set_axis_off()
    ax.set_aspect(1)
    plt.ioff()
    # plt.axis('off')
    # plt.gca().set_aspect(1)
    plt.show()

    # qApp.exec_()
    # while True:
    #     time.sleep(0.1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--duration', help='Input duration (secs) to select from input file [10.0]',
                        default=10.0, type=float)
    parser.add_argument('-f', '--file', help='Sound file to process', default=None, type=str)

    args = parser.parse_args()

    main_graph(args)
