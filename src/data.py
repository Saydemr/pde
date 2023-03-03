"""
Code partially copied from 'Diffusion Improves Graph Learning' repo https://github.com/klicperajo/gdc/blob/master/data.py
"""

import os

import numpy as np
import copy

import torch
from eppugnn import Eppugnn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from graph_rewiring import get_two_hop, apply_gdc
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from graph_rewiring import make_symmetric, apply_pos_dist_rewire
from heterophilic import WebKB, WikipediaNetwork, Actor
from utils import ROOT_DIR

DATA_PATH = f'{ROOT_DIR}/data'


def rewire(data, opt, data_dir):
  rw = opt['rewiring']
  if rw == 'two_hop':
    data = get_two_hop(data)
  elif rw == 'gdc':
    data = apply_gdc(data, opt)
  elif rw == 'pos_enc_knn':
    data = apply_pos_dist_rewire(data, opt, data_dir)
  return data


def get_dataset(opt: dict, data_dir, use_lcc: bool = False) -> InMemoryDataset:
  ds = opt['dataset']
  path = os.path.join(data_dir, ds)
  if ds in ['dm', 'mm', 'hs', 'sc']:
    dataset = Eppugnn(path, ds, transform=T.NormalizeFeatures())
  elif ds in ['Cora', 'Citeseer', 'Pubmed']:
    dataset = Planetoid(path, ds)
  elif ds in ['Computers', 'Photo']:
    dataset = Amazon(path, ds)
  elif ds == 'CoauthorCS':
    dataset = Coauthor(path, 'CS')
  elif ds in ['cornell', 'texas', 'wisconsin']:
    dataset = WebKB(root=path, name=ds, transform=T.NormalizeFeatures())
  elif ds in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(root=path, name=ds, transform=T.NormalizeFeatures())
  elif ds == 'film':
    dataset = Actor(root=path, transform=T.NormalizeFeatures())
  elif ds == 'ogbn-arxiv':
    dataset = PygNodePropPredDataset(name=ds, root=path,
                                     transform=T.ToSparseTensor())
    use_lcc = False  # never need to calculate the lcc with ogb datasets
  else:
    raise Exception('Unknown dataset.')

  if opt['rewiring'] is not None:
    dataset.data = rewire(dataset.data, opt, data_dir)
  train_mask_exists = True
  try:
    dataset.data.train_mask
  except AttributeError:
    train_mask_exists = False

  # dim reduction & oversampling
  dataset.data = sample_augment(12345, dataset.data)

  return dataset


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = dataset.data.edge_index.numpy()
  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
  remaining_nodes = set(range(dataset.data.x.shape[0]))
  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(dataset, start)
    comps.append(comp)
    remaining_nodes = remaining_nodes.difference(comp)
  return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 1
  return mapper


def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]


def sample_augment(
        seed: int,
        data: Data) -> Data:
  
  # reduce dimensionality of node features with PCA
  from sklearn.decomposition import PCA
  pca = PCA(n_components=64)
  data.x = torch.from_numpy(pca.fit_transform(data.x.numpy()))

  # get the train data and oversample it to have equal number of instances from each class
  train_data = data.x[data.train_mask]
  train_labels = data.y[data.train_mask]
  train_len = len(train_data)
  rnd_state = np.random.RandomState(seed)


  # oversample the minority class
  from imblearn.over_sampling import SVMSMOTE
  ros = SVMSMOTE(random_state=rnd_state, sampling_strategy='minority')
  train_data, train_labels = ros.fit_resample(train_data, train_labels)
  train_data = torch.from_numpy(train_data)
  train_labels = torch.from_numpy(train_labels)

  train_orig = train_data[:train_len]
  train_aug = train_data[train_len:]

  # find nodes with label 1 in train_orig
  train_orig_1 = train_orig[train_labels[:train_len] == 1]

  counter = 0
  init_edges = copy.deepcopy(data.edge_index.numpy())
  for instance in train_aug:
    # get the most similar node in the original training set
    dists = torch.cdist(instance.unsqueeze(0), train_orig_1)
    closest_node = torch.argmin(dists, dim=1)
    closest_node = closest_node.item()
    # get the edges of the closest node
    row, col = init_edges
    closest_node_edges = np.where(row == closest_node)[0]
    closest_node_edges = [(row[i], col[i]) for i in closest_node_edges]
    # add the edges to the augmented node
    for e in closest_node_edges:
      data.edge_index = torch.cat((data.edge_index, torch.tensor(e).unsqueeze(1)), dim=1)
    
    counter += 1


  val_data = data.x[data.val_mask]
  val_labels = data.y[data.val_mask]

  test_data = data.x[data.test_mask]
  test_labels = data.y[data.test_mask]

  # combine the train, val, and test data
  data.x = torch.cat((train_data, val_data, test_data), dim=0)
  data.y = torch.cat((train_labels, val_labels, test_labels), dim=0)

  # get the indices of the train, val, and test data
  train_idx = np.arange(train_data.shape[0])
  val_idx = np.arange(train_data.shape[0], train_data.shape[0] + val_data.shape[0])
  test_idx = np.arange(train_data.shape[0] + val_data.shape[0], data.x.shape[0])


  # set the train, val, and test masks
  data.train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
  data.train_mask[train_idx] = True
  data.val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
  data.val_mask[val_idx] = True
  data.test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
  data.test_mask[test_idx] = True

  return data
