import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


class ServiceGraph:

    def __init__(self):
        self._digraph = None
        self._graph_df = None

    def loadFromFile(self, file_path: str):
        df = pd.read_csv(file_path)
        self._graph_df = df

        return self

    def load(self):
        if self._graph_df is not None:
            DG = nx.DiGraph()
            for index, row in self._graph_df.iterrows():
                source = row['source']
                destination = row['destination']
                if 'rabbitmq' not in source and 'rabbitmq' not in destination and 'db' not in destination and 'db' not in source:
                    DG.add_edge(source, destination)

            for node in DG.nodes():
                if 'kubernetes' in node:
                    DG.nodes[node]['type'] = 'host'
                else:
                    DG.nodes[node]['type'] = 'service'

            self._digraph = DG
        else:
            raise Exception("Haven't cached the dependency graph data.")
        return self

    def showGraph(self):
        plt.figure(figsize=(9, 9))
        nx.draw(self._digraph, with_labels=True, font_weight='bold')
        pos = nx.spring_layout(self._digraph)
        nx.draw(self._digraph, pos, with_labels=True, cmap=plt.get_cmap('jet'), node_size=1500, arrows=True, )
        labels = nx.get_edge_attributes(self._digraph, 'weight')
        nx.draw_networkx_edge_labels(self._digraph, pos, edge_labels=labels)
        plt.show()

    @property
    def digraph(self):
        return self._digraph
