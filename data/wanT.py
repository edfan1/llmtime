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


def get_want_dataset(n=-1,testfrac=0.15, predict_steps=1000, egress = 'SACR', ingress = 'SUNN'):
    datasets = []
    datas = []
    dg = readData("./datasets/wanT/snmp_2018_1hourinterval.csv")
    edge = dg.get_edge_data(ingress, egress)['data']
    edge_name = dg.get_edge_data(ingress, egress)['name']
    datasets.append(edge_name)
    # splitpoint = len(edge) - predict_steps
    splitpoint = int(len(edge)*(1-testfrac))
    # train = edge.iloc[:splitpoint]['SACR_SUNN_out']
    # test = edge.iloc[splitpoint:]['SACR_SUNN_out']
    scaler = StandardScaler()
    # print(scaler.fit(edge['SACR_SUNN_out'].to_frame()))
    # print(scaler.mean_)
    scaled_data = pd.DataFrame(
        np.round(
            scaler.fit_transform(edge['SACR_SUNN_out'].to_frame()),
              4))
    print(scaled_data)
    print(scaler.mean_)
    train = scaled_data.iloc[:splitpoint]
    test = scaled_data.iloc[splitpoint:]
    datas.append((train, test))
    return dict(zip(datasets,datas)), scaler

def get_scaled_dataset(n=-1,testfrac=0.15, predict_steps=1000, egress = 'SACR', ingress = 'SUNN'):
    datasets = []
    datas = []
    dg = readData("./datasets/wanT/snmp_2018_1hourinterval.csv")
    edge = dg.get_edge_data(ingress, egress)['data']
    edge_name = dg.get_edge_data(ingress, egress)['name']
    datasets.append(edge_name)
    # splitpoint = len(edge) - predict_steps
    splitpoint = int(len(edge)*(1-testfrac))
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        np.round(
            scaler.fit_transform(edge['SACR_SUNN_out'].to_frame()),
              4))[0]
    print(scaled_data)
    print(scaler.mean_)
    train = scaled_data.iloc[:splitpoint]
    test = scaled_data.iloc[splitpoint:]
    datas.append((train, test))
    return dict(zip(datasets,datas)), scaler

def main():
    x = get_scaled_dataset()
    # print(x)

if __name__ == '__main__':
    main()