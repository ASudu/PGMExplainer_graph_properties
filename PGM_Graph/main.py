import os
import numpy as np
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader
import networkx as nx
from torchvision import datasets as ds
from torchvision import transforms
import argparse
import pickle as pkl
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
from nets.superpixels_graph_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset
from utils import GCN_params
import pgm_explainer_graph as pe # import explainer

def _get_train_val_test_masks(total_size, y_true, val_fraction, test_fraction, seed):
    """Performs stratified train/test/val split

    Args:
        total_size (int): dataset total number of instances
        y_true (numpy array): labels
        val_fraction (int): validation/test set proportion
        test_fraction (int): test and val sets proportion
        seed (int): seed value

    Returns:
        [torch.tensors]: train, validation and test masks - boolean values
    """
    # Split into a train, val and test set
    # Store indexes of the nodes belong to train, val and test set
    indexes = range(total_size)
    indexes_train, indexes_test = train_test_split(
        indexes, test_size=test_fraction, stratify=y_true, random_state=seed)
    indexes_train, indexes_val = train_test_split(indexes_train, test_size=val_fraction, stratify=y_true[indexes_train],
                                                  random_state=seed)
    # Init masks
    train_idxs = np.zeros(total_size, dtype=np.bool)
    val_idxs = np.zeros(total_size, dtype=bool)
    test_idxs = np.zeros(total_size, dtype=np.bool)

    # Update masks using corresponding indexes
    train_idxs[indexes_train] = True
    val_idxs[indexes_val] = True
    test_idxs[indexes_test] = True

    return torch.from_numpy(train_idxs), torch.from_numpy(val_idxs), torch.from_numpy(test_idxs)

def split_function(y, args_train_ratio=0.6, seed=10):
    return _get_train_val_test_masks(y.shape[0], y, (1-args_train_ratio)/2, (1-args_train_ratio), seed=seed)

def arg_parse():
    parser = argparse.ArgumentParser(description="PGM Explainer arguments.")
    parser.add_argument(
            "--start", dest="start", type=int, help="Index of starting image."
        )
    parser.add_argument(
            "--end", dest="end", type=int, help="Index of ending image."
        )
    parser.add_argument(
            "--dataset", dest="dataset", help="Dataset to use for the graph classification task."
        )
    parser.add_argument("--perturb-indicator", dest="perturb_indicator", help="diff or abs.")
    parser.add_argument("--perturb-mode", dest="perturb_mode", help="mean, zero, max or uniform.")
    parser.add_argument("--perturb-feature", dest="perturb_feature", help="color or location.")
    
    parser.set_defaults(
        start = 0,
        end = 100,
        dataset = "MNIST",
        perturb_indicator = "diff",
        perturb_mode = "mean",
        perturb_feature = "color"
    )
    return parser.parse_args()

prog_args = arg_parse()

MODEL_NAME = 'GCN'
if prog_args.dataset == "MNIST":
    MNIST_test_dataset = ds.MNIST(root='PATH', train=False, download=True, transform=transforms.ToTensor())
    DATASET_NAME = 'MNIST'
    dataset = LoadData(DATASET_NAME)
    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    net_params = GCN_params.net_params()
    model = gnn_model(MODEL_NAME, net_params)
    model.load_state_dict(torch.load("data/superpixels/epoch_188.pkl"))
    model.eval()

    test_loader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=False, collate_fn=dataset.collate)

elif prog_args.dataset == "syn6":
    # Define path where dataset should be saved
    data_path = "data/{}.pth".format("syn6")

    # If already created, do not recreate
    if os.path.exists(data_path):
        data = torch.load(data_path)
    
    else:
        data = SimpleNamespace()
        with open('data/syn6/BA-2motif.pkl', 'rb') as fin:
            data.edge_index, data.x, data.y = pkl.load(fin)
        data.x = np.ones_like(data.x)

    # Define NumSpace dataset
    data.x = torch.FloatTensor(data.x)
    data.edge_index = torch.FloatTensor(data.edge_index)
    data.y = torch.LongTensor(data.y)
    _, data.y = data.y.max(dim=-1)
    data.num_classes = 2
    data.num_features = data.x.shape[-1]
    data.num_nodes = data.edge_index.shape[1]
    data.num_graphs = data.x.shape[0]
    data.name = prog_args.dataset

    # Shuffle graphs 
    p = torch.randperm(data.num_graphs)
    data.x = data.x[p]
    data.y = data.y[p]
    data.edge_index = data.edge_index[p]
    
    # Train / Val / Test split
    data.train_mask, data.val_mask, data.test_mask = split_function(data.y, 0.8)    # [train_ration, val_ration, test_ratio] = [0.8,0.1,0.1]
    # Save data
    torch.save(data, data_path)

    net_params = GCN_params.net_params()
    model = gnn_model(MODEL_NAME, net_params)
    model.load_state_dict(torch.load("data/superpixels/epoch_188.pkl"))
    model.eval()

    test_loader = DataLoader(data.test_mask, batch_size=1, shuffle=False, drop_last=False)

index_to_explain = range(prog_args.start, prog_args.end)
if prog_args.perturb_feature == "color":
    perturb_features_list = [0]
elif prog_args.perturb_feature == "location":
    perturb_features_list = [1,2]


Explanations = []
for iter, (graph, label, snorm_n, snorm_e) in enumerate(test_loader):
    if iter in index_to_explain:
        pred = model.forward(graph, graph.ndata['feat'],graph.edata['feat'],snorm_n, snorm_e)
        soft_pred = np.asarray(softmax(np.asarray(pred[0].data)))
        pred_threshold = 0.1*np.max(soft_pred)
        e = pe.Graph_Explainer(model, graph, 
                               snorm_n = snorm_n, snorm_e = snorm_n, 
                               perturb_feature_list = perturb_features_list,
                               perturb_mode = prog_args.perturb_mode,
                               perturb_indicator = prog_args.perturb_indicator)
        pgm_nodes, p_values, candidates = e.explain(num_samples = 1000, percentage = 10, 
                                top_node = 4, p_threshold = 0.05, pred_threshold = pred_threshold)
        label = np.argmax(soft_pred)
        pgm_nodes_filter = [i for i in pgm_nodes if p_values[i] < 0.02]
        x_cor = [e.X_feat[node_][1] for node_ in pgm_nodes_filter]
        y_cor = [e.X_feat[node_][2] for node_ in pgm_nodes_filter]
        result = [iter, label, pgm_nodes_filter, x_cor, y_cor]
        print(result)
        Explanations.append(result)
        savedir = 'result/explanations_'+ str(prog_args.start) + "_" + str(prog_args.end) +".txt"
        with open(savedir, "a") as text_file:
            text_file.write(str(result) + "\n")
            
            
            
            