import numpy as np
import pandas as pd
import networkx as nx

from sklearn.cluster import Birch
from sklearn import preprocessing

from rca.microRCA.metrics import ServiceMetrics, HostMetrics
from rca.microRCA.svcGraph import ServiceGraph


class MicroRCA:
    """
    READ:https://github.com/elastisys/MicroRCA
    """

    def __init__(self, alpha=0.55, ad_threshold=0.045, smoothing_window=12, front_end_svc_names=['front-end']):
        self.alpha = alpha
        self.ad_threshold = ad_threshold
        self.smoothing_window = smoothing_window

        self.front_end_svc_names = front_end_svc_names

        self.svc_metrics_df = None
        self.host_metrics_df = None
        self.dependency_graph = None

    def searchRootCause(self, svc_metrics: ServiceMetrics, host_metrics: HostMetrics,
                        dependency_graph: ServiceGraph, target, fault_type):

        # Step1: Initialize the environment
        # load the metrics data and dependency graph
        # TODO: load data from OLAP
        self.svc_metrics_df = svc_metrics.load().data
        self.host_metrics_df = host_metrics.load().data
        self.dependency_graph = dependency_graph

        print('start to search Root Cause')
        print('target:', target, ' fault_type:', fault_type)

        # anomaly detection on response time of service invocation
        anomalies = self._birch_ad_with_smoothing(self.svc_metrics_df)
        print("find the anomalies: \n{}".format(anomalies))

        # Step2: Attributed graph
        # construct attributed graph
        # TODO: Load the graph from the GRAPH database (neo4j)
        DG = self._attributed_graph()

        # Step3: Anomalous subgraph & Step4: Weighted and Scored Anomalous Subgraph
        anomaly_graph, personalization = self._anomaly_subgraph(anomalies, DG)

        # Step4: Localizing Faulty Services
        anomaly_score = self._calc_and_rank_root_cause_scores(anomaly_graph, personalization)
        print("the root cause scores: {}".format(anomaly_score))
        return anomaly_score

    def _calc_and_rank_root_cause_scores(self, anomaly_graph, personalization):
        anomaly_graph = anomaly_graph.reverse(copy=True)
        edges = list(anomaly_graph.edges(data=True))
        for u, v, d in edges:
            if anomaly_graph.nodes[v]['type'] == 'host':
                anomaly_graph.remove_edge(u, v)
                anomaly_graph.add_edge(v, u, weight=d['weight'])
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(9,9))
        # nx.draw(DG, with_labels=True, font_weight='bold')
        # pos = nx.spring_layout(anomaly_graph)
        # nx.draw(anomaly_graph, pos, with_labels=True, cmap = plt.get_cmap('jet'), node_size=1500, arrows=True, )
        # labels = nx.get_edge_attributes(anomaly_graph,'weight')
        # nx.draw_networkx_edge_labels(anomaly_graph,pos,edge_labels=labels)
        # plt.show()
        anomaly_score = nx.pagerank(anomaly_graph, alpha=0.85, personalization=personalization, max_iter=10000)
        anomaly_score = sorted(anomaly_score.items(), key=lambda x: x[1], reverse=True)
        return anomaly_score

    def _birch_ad_with_smoothing(self, latency_df):
        threshold = self.ad_threshold
        smoothing_window = self.smoothing_window
        anomalies = []

        for svc, latency in latency_df.iteritems():
            # No anomaly detection in db
            if svc != 'timestamp' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
                latency = latency.rolling(window=smoothing_window, min_periods=1).mean()
                x = np.array(latency)
                x = np.where(np.isnan(x), 0, x)
                normalized_x = preprocessing.normalize([x])

                X = normalized_x.reshape(-1, 1)

                brc = Birch(branching_factor=50, n_clusters=None, threshold=threshold, compute_labels=True)
                brc.fit(X)
                brc.predict(X)

                labels = brc.labels_
                # centroids = brc.subcluster_centers_
                n_clusters = np.unique(labels).size
                if n_clusters > 1:
                    anomalies.append(svc)
        return anomalies

    def _attributed_graph(self):
        self.dependency_graph.load()
        return self.dependency_graph.digraph

    def _anomaly_subgraph(self, anomalies, DG):

        # all the anomaly and adjacent nodes' edges
        edges = []
        # all the anomaly and adjacent node
        nodes = []
        # the rt_a_j value
        baseline_df = pd.DataFrame()

        # extract anomaly node from the anomalies edges
        for anomaly in anomalies:
            edge = anomaly.split('_')
            edges.append(tuple(edge))
            svc_name = edge[1]
            nodes.append(svc_name)
            if svc_name in baseline_df.columns:
                baseline_df[svc_name] += self.svc_metrics_df[anomaly]
            else:
                baseline_df[svc_name] = self.svc_metrics_df[anomaly]
        # remove the duplicated node
        nodes = set(nodes)
        print("find the anomlay node: \n {}".format(nodes))

        # init personalization through the dependency graph's nodes
        personalization = {}
        for node in DG.nodes():
            if node in nodes:
                personalization[node] = 0

        # Get the subgraph of anomaly and calculate the weight
        # Anomalous Subgraph Extraction & Location Fault Service
        anomaly_graph = nx.DiGraph()
        self._locate_fault_service(DG, anomaly_graph, baseline_df, edges, nodes, personalization)

        return anomaly_graph, personalization

    def _locate_fault_service(self, DG, anomaly_graph, baseline_df, edges, nodes, personalization):
        # Anomalous Subgraph Weighing
        # for node of anomaly nodes
        for node in nodes:
            # for edge e_ij in-edges of vj(node)
            for u, v, data in DG.in_edges(node, data=True):
                edge = (u, v)
                # if edge in anomaly edges or not, calculate the weight
                # see the formula in the MicroRCA paper
                if edge in edges:
                    data = self.alpha
                else:
                    normal_edge = u + '_' + v
                    data = baseline_df[v].corr(self.svc_metrics_df[normal_edge])

                data = round(data, 3)
                anomaly_graph.add_edge(u, v, weight=data)
                anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
                anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']
            # Set personalization with container resource usage
            # for edge e_jk out-edges of vj(node)
            for u, v, data in DG.out_edges(node, data=True):
                edge = (u, v)
                # if nodes don't contains the start node of the anomaly edges, it's always False
                if edge in edges:
                    data = alpha
                elif DG.nodes[v]['type'] == 'host':
                    data, col = self._cals_node_corr_host_weight(u, anomaly_graph, baseline_df)
                else:
                    normal_edge = u + '_' + v
                    data = baseline_df[u].corr(self.svc_metrics_df[normal_edge])
                data = round(data, 3)
                anomaly_graph.add_edge(u, v, weight=data)
                anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
                anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

        # Assigning Service Anomaly Score
        # for node of anomaly nodes
        for node in nodes:
            # calc Service Anomaly Score => personalization
            # compute anomaly score(AS) for anomalous service nodes
            personalization[node] = self._calc_node_anomaly_scores(node, anomaly_graph, baseline_df)
        print("personalization: {}".format(personalization))

    def _cals_node_corr_host_weight(self, svc, anomaly_graph, baseline_df):

        node_cols = ['node_cpu', 'node_network', 'node_memory']
        max_corr = 0.01
        metric = 'node_cpu'
        for col in node_cols:
            temp = abs(baseline_df[svc].corr(self.host_metrics_df[col]))
            if temp > max_corr:
                max_corr = temp
                metric = col

        # Get the average weight of the in_edges
        in_edges_weight_avg = 0.0
        num = 0
        for u, v, data in anomaly_graph.in_edges(svc, data=True):
            #        print(u, v)
            num = num + 1
            in_edges_weight_avg = in_edges_weight_avg + data['weight']
        if num > 0:
            in_edges_weight_avg = in_edges_weight_avg / num

        data = in_edges_weight_avg * max_corr
        return data, metric

    def _calc_node_anomaly_scores(self, svc, anomaly_graph, baseline_df):
        df = self.host_metrics_df

        ctn_cols = ['ctn_cpu', 'ctn_network', 'ctn_memory']
        max_corr = 0.01
        metric = 'ctn_cpu'
        for col in ctn_cols:
            temp = abs(baseline_df[svc].corr(df[col]))
            if temp > max_corr:
                max_corr = temp
                metric = col

        edges_weight_avg = 0.0
        num = 0
        for u, v, data in anomaly_graph.in_edges(svc, data=True):
            num = num + 1
            edges_weight_avg = edges_weight_avg + data['weight']

        for u, v, data in anomaly_graph.out_edges(svc, data=True):
            if anomaly_graph.nodes[v]['type'] == 'service':
                num = num + 1
                edges_weight_avg = edges_weight_avg + data['weight']

        edges_weight_avg = edges_weight_avg / num

        personalization = edges_weight_avg * max_corr

        return personalization / anomaly_graph.degree(svc)


if __name__ == '__main__':
    # Tuning parameters
    alpha = 0.55
    ad_threshold = 0.045
    smoothing_window = 12

    # faults_type = ['svc_latency', 'service_cpu', 'service_memory']
    faults_type = ['svc_latency']
    # targets = ['front-end', 'catalogue', 'orders', 'user', 'carts', 'payment', 'shipping']
    targets = ['front-end']
    # load the metrics data（latency）
    svcMetrics = ServiceMetrics()
    svcMetrics.load_from_file("/Users/Renzhiqiang/Workspace/data/rca/simple/service_cpu_carts_latency_source_90.csv")
    svcMetrics.data["unknown_front-end"] = 0
    svcMetrics.load_from_file(
        "/Users/Renzhiqiang/Workspace/data/rca/simple/service_cpu_carts_latency_destination_50.csv")

    # load the host metrics
    hostMetrics = HostMetrics()
    hostMetrics.load_from_file("/Users/Renzhiqiang/Workspace/data/rca/simple/service_cpu_carts_user.csv")

    # construct service graph
    svcGraph = ServiceGraph()
    svcGraph.loadFromFile("/Users/Renzhiqiang/Workspace/data/rca/simple/service_cpu_carts_mpg.csv")

    # construct model
    model = MicroRCA()
    for fault_type in faults_type:
        for target in targets:
            if target == 'front-end' and fault_type != 'svc_latency':
                # skip front-end for service_latency(cpu & memory)
                continue

            print('target:', target, ' fault_type:', fault_type)
            model.searchRootCause(svcMetrics, hostMetrics, svcGraph, target, fault_type)
