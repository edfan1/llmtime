import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import datasets
from datasets import load_dataset
import os
import pickle

def readData(fileName, timeCol=0):
    DG = nx.DiGraph()
    data = pd.read_csv(fileName)
    linkname = data.columns
    # print(linkname)
    time = data.columns[timeCol]
    #data[time] = pd.to_datetime(data[time])
    #data[time] = time
    for head in linkname[(timeCol + 1):]:
        arr = head.split("_")
        if len(arr) == 3 and arr[2] == "in":
            DG.add_edge(arr[0], arr[1], data=pd.DataFrame(data, columns=[time, head]))
            edge_name = arr[0] + "_" + arr[1]
            DG[arr[0]][arr[1]]['name'] = edge_name
        else:
            DG.add_edge(arr[1], arr[0], data=pd.DataFrame(data, columns=[time, head]))
            edge_name = arr[1] + "_" + arr[0]
            DG[arr[1]][arr[0]]['name'] = edge_name
    print(DG.edges)
    return DG


def get_want_dataset(n=-1,testfrac=0.15, predict_steps=1000, ingress = 'SCAR', egress = 'SUNN'):
    datasets = []
    datas = []
    dg = readData("./datasets/wanT/snmp_2018_1hourinterval.csv")
    edge = dg.get_edge_data(ingress, egress)['data']
    edge_name = dg.get_edge_data(ingress, egress)['name']
    datasets.append(edge_name)
    splitpoint = len(edge) - predict_steps
    train = edge.iloc[:splitpoint]
    test = edge.iloc[splitpoint:]
    datas.append((train, test))
    return dict(zip(datasets,datas))