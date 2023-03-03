import argparse
import os
from functools import partial

import numpy as np
import torch
from data import get_dataset, sample_augment
from GNN_early import GNNEarly
from GNN import GNN
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from run_GNN import get_optimizer, test, test_OGB, train
from torch import nn
from CGNN import CGNN, get_sym_adj
from CGNN import train as train_cgnn

"""
python3 ray_tune.py --dataset sc --lr 0.001 --add_source --function transformer --attention_dim 16 --hidden_dim 64 --heads 4 --input_dropout 0 --decay 0 --adjoint --epoch 100 --num_samples 1 --name sc-test --gpus 1 --grace_period 50 --cpus 8 --block transformer --regularise --use_labels --label_rate 1
"""


def average_test(models, datas):
    if opt['dataset'] == 'ogbn-arxiv':
        results = [test_OGB(model, data, opt) for model, data in zip(models, datas)]
    else:
        results = [test(model, data) for model, data in zip(models, datas)]
    train_accs, val_accs, tmp_test_accs = [], [], []

    for train_acc, val_acc, test_acc in results:
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        tmp_test_accs.append(test_acc)

    return train_accs, val_accs, tmp_test_accs


def train_ray_rand(opt, checkpoint_dir=None, data_dir="../data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(opt, data_dir, opt['not_lcc'])
    models = []
    datas = []
    optimizers = []

    for split in range(opt["num_splits"]):
        dataset.data = sample_augment(
            np.random.randint(0, 1000), dataset.data)
        datas.append(dataset.data)

        if opt['baseline']:
            opt['num_feature'] = dataset.num_node_features
            opt['num_class'] = dataset.num_classes
            adj = get_sym_adj(dataset.data, opt, device)
            model, data = CGNN(opt, adj, opt['time'], device).to(device), dataset.data.to(device)
            train_this = train_cgnn
        else:
            model = GNN(opt, dataset, device)
            train_this = train

        models.append(model)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model, data = model.to(device), dataset.data.to(device)
        parameters = [p for p in model.parameters() if p.requires_grad]

        optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])
        optimizers.append(optimizer)

        # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
        # should be restored.
        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    for epoch in range(1, opt["epoch"]):
        loss = np.mean(
            [train_this(model, optimizer, data) for model, optimizer, data in zip(models, optimizers, datas)])
        train_accs, val_accs, tmp_test_accs = average_test(models, datas)
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            best = np.argmax(val_accs)
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((models[best].state_dict(), optimizers[best].state_dict()), path)
        tune.report(loss=loss, accuracy=np.mean(val_accs), test_acc=np.mean(tmp_test_accs),
                    train_acc=np.mean(train_accs),
                    forward_nfe=model.fm.sum,
                    backward_nfe=model.bm.sum)


def train_ray(opt, checkpoint_dir=None, data_dir="../data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(opt, data_dir, opt['not_lcc'])

    models = []
    optimizers = []

    data = dataset.data.to(device)
    datas = [data for i in range(opt["num_init"])]

    for split in range(opt["num_init"]):
        if opt['baseline']:
            opt['num_feature'] = dataset.num_node_features
            opt['num_class'] = dataset.num_classes
            adj = get_sym_adj(dataset.data, opt, device)
            model, data = CGNN(opt, adj, opt['time'], device).to(device), dataset.data.to(device)
            train_this = train_cgnn
        else:
            model = GNN(opt, dataset, device)
            train_this = train

        models.append(model)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.to(device)
        parameters = [p for p in model.parameters() if p.requires_grad]

        optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])
        optimizers.append(optimizer)

        # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
        # should be restored.
        if checkpoint_dir:
            checkpoint = os.path.join(checkpoint_dir, "checkpoint")
            model_state, optimizer_state = torch.load(checkpoint)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    for epoch in range(1, opt["epoch"]):
        loss = np.mean([train_this(model, optimizer, data) for model, optimizer in zip(models, optimizers)])
        train_accs, val_accs, tmp_test_accs = average_test(models, datas)
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            best = np.argmax(val_accs)
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((models[best].state_dict(), optimizers[best].state_dict()), path)
        tune.report(loss=loss, accuracy=np.mean(val_accs), test_acc=np.mean(tmp_test_accs),
                    train_acc=np.mean(train_accs),
                    forward_nfe=model.fm.sum,
                    backward_nfe=model.bm.sum)


def train_ray_int(opt, checkpoint_dir=None, data_dir="../data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    dataset = get_dataset(opt, data_dir, opt['not_lcc'])

    if opt["num_splits"] > 0:
        dataset.data = sample_augment(
            23 * np.random.randint(0, opt["num_splits"]),
            # random prime 23 to make the splits 'more' random. Could remove
            dataset.data)

    model = GNN(opt, dataset, device) if opt["no_early"] else GNNEarly(opt, dataset, device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model, data = model.to(device), dataset.data.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(opt["optimizer"], parameters, lr=opt["lr"], weight_decay=opt["decay"])

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    this_test = test_OGB if opt['dataset'] == 'ogbn-arxiv' else test
    best_time = best_epoch = train_acc = val_acc = test_acc = 0
    for epoch in range(1, opt["epoch"]):
        loss = train(model, optimizer, data)
        # need next line as it sets the attributes in the solver

        if opt["no_early"]:
            tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, opt)
            best_time = opt['time']
        else:
            tmp_train_acc, tmp_val_acc, tmp_test_acc = this_test(model, data, opt)
        if tmp_val_acc > val_acc:
            best_epoch = epoch
            train_acc = tmp_train_acc
            val_acc = tmp_val_acc
            test_acc = tmp_test_acc
        if model.odeblock.test_integrator.solver.best_val > val_acc:
            best_epoch = epoch
            val_acc = model.odeblock.test_integrator.solver.best_val
            test_acc = model.odeblock.test_integrator.solver.best_test
            train_acc = model.odeblock.test_integrator.solver.best_train
            best_time = model.odeblock.test_integrator.solver.best_time
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=loss, accuracy=val_acc, test_acc=test_acc, train_acc=train_acc, best_time=best_time,
                    best_epoch=best_epoch,
                    forward_nfe=model.fm.sum, backward_nfe=model.bm.sum)



def set_eppugnn_search_space(opt):
    opt["decay"] = tune.loguniform(0.001, 0.1)  # weight decay l2 reg
    if opt['regularise']:
        opt["kinetic_energy"] = tune.loguniform(0.001, 10.0)
        opt["directional_penalty"] = tune.loguniform(0.001, 10.0)

    opt["hidden_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(6, 8))  # hidden dim of X in dX/dt
    opt["lr"] = tune.uniform(0.01, 0.2)
    # opt["input_dropout"] = tune.uniform(0.2, 0.8)  # encoder dropout
    opt["input_dropout"] = 0.5
    opt["optimizer"] = tune.choice(["adam", "adamax"])
    opt["dropout"] = tune.uniform(0, 0.15)  # output dropout
    opt["time"] = tune.uniform(2.0, 30.0)  # terminal time of the ODE integrator;
    # when it's big, the training hangs (probably due a big NFEs of the ODE)

    if opt["block"] in {'attention', 'mixed'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
        opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(0, 4))  #
        opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))  # hidden dim for attention
        opt['attention_norm_idx'] = tune.choice([0, 1])
        # opt['attention_norm_idx'] = 0
        opt["leaky_relu_slope"] = tune.uniform(0, 0.7)
        # opt["leaky_relu_slope"] = 0.2

        opt["self_loop_weight"] = tune.choice([0, 1])  # whether or not to use self-loops
    else:
        opt["self_loop_weight"] = tune.uniform(0, 3)

    opt["tol_scale"] = tune.loguniform(1, 1000)  # num you multiply the default rtol and atol by
    if opt["adjoint"]:
        opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun"])  # , "rk4"])
        opt["tol_scale_adjoint"] = tune.loguniform(100, 10000)

    opt['add_source'] = tune.choice([True, False])
    opt['att_samp_pct'] = tune.uniform(0.3, 1)
    opt['batch_norm'] = tune.choice([True, False])
    # opt['batch_norm'] = True

    if opt['rewiring'] == 'gdc':
        opt['gdc_k'] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 10))
        opt['ppr_alpha'] = tune.uniform(0.01, 0.2)

        # opt['gdc_sparsification'] = tune.choice(['topk', 'threshold'])
        opt['gdc_sparsification'] = 'threshold'
        opt['exact'] = False
        # opt['gdc_method'] = tune.choice(['ppr', 'heat'])
        opt['gdc_method'] = 'ppr'
        opt['avg_degree'] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))  #  bug currently in pyg
        opt['gdc_threshold'] = tune.loguniform(0.00001, 0.01)
        # opt['gdc_threshold'] = None
        opt['ppr_alpha'] = tune.uniform(0.01, 0.2)
        # opt['heat_time'] = tune.uniform(1, 5)

    return opt






def set_search_space(opt):
    if opt["dataset"] in ["dm", "mm", "sc", "hs"]:
        return set_eppugnn_search_space(opt)




def main(opt):
    data_dir = os.path.abspath("../data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = set_search_space(opt)
    scheduler = ASHAScheduler(
        metric=opt['metric'],
        mode="max",
        max_t=opt["epoch"],
        grace_period=opt["grace_period"],
        reduction_factor=opt["reduction_factor"],
    )
    reporter = CLIReporter(
        metric_columns=["accuracy", "test_acc", "train_acc", "loss", "training_iteration", "forward_nfe",
                        "backward_nfe"]
    )
    # choose a search algorithm from https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
    search_alg = None

    train_fn = train_ray if opt["num_splits"] == 0 else train_ray_rand

    result = tune.run(
        partial(train_fn, data_dir=data_dir),
        name=opt["name"],
        resources_per_trial={"cpu": opt["cpus"], "gpu": opt["gpus"]},
        search_alg=search_alg,
        keep_checkpoints_num=3,
        checkpoint_score_attr=opt['metric'],
        config=opt,
        num_samples=opt["num_samples"],
        scheduler=scheduler,
        max_failures=2,
        local_dir="../ray_tune",
        progress_reporter=reporter,
        raise_on_failed_trial=False,
        resume=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_cora_defaults",
        action="store_true",
        help="Whether to run with best params for cora. Overrides the choice of dataset",
    )
    parser.add_argument(
        "--dataset", type=str, default="Cora", help="Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS"
    )
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension.")
    parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                        help='Add a fully connected layer to the decoder.')
    parser.add_argument("--input_dropout", type=float, default=0.5, help="Input dropout rate.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--decay", type=float, default=5e-4, help="Weight decay for optimization")
    parser.add_argument("--self_loop_weight", type=float, default=1.0, help="Weight of self-loops.")
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--label_rate', type=float, default=1,
                        help='% of training labels to use when --use_labels is set.')
    parser.add_argument("--epoch", type=int, default=100, help="Number of training epochs per iteration.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Factor in front matrix A.")
    parser.add_argument("--time", type=float, default=1.0, help="End time of ODE function.")
    parser.add_argument("--augment", action="store_true",
                        help="double the length of the feature vector by appending zeros to stabilise ODE learning", )
    parser.add_argument("--alpha_dim", type=str, default="sc", help="choose either scalar (sc) or vector (vc) alpha")
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument("--beta_dim", type=str, default="sc", help="choose either scalar (sc) or vector (vc) beta")
    parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                        help='Add a fully connected layer to the encoder.')
    # ODE args
    parser.add_argument(
        "--method", type=str, default="dopri5", help="set the numerical solver: dopri5, euler, rk4, midpoint"
    )
    parser.add_argument('--step_size', type=float, default=1,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument(
        "--adjoint_method", type=str, default="adaptive_heun",
        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint"
    )
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument("--adjoint", dest='adjoint', action='store_true',
                        help="use the adjoint ODE method to reduce memory footprint")
    parser.add_argument("--tol_scale", type=float, default=1.0, help="multiplier for atol and rtol")
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument("--ode_blocks", type=int, default=1, help="number of ode blocks to run")
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--add_source', dest='add_source', action='store_true',
                        help='If try get rid of alpha param and the beta*x0 source term')
    # SDE args
    parser.add_argument("--dt_min", type=float, default=1e-5, help="minimum timestep for the SDE solver")
    parser.add_argument("--dt", type=float, default=1e-3, help="fixed step size")
    parser.add_argument('--adaptive', dest='adaptive', action='store_true', help='use adaptive step sizes')
    # Attention args
    parser.add_argument(
        "--leaky_relu_slope",
        type=float,
        default=0.2,
        help="slope of the negative part of the leaky relu used in attention",
    )
    parser.add_argument('--attention_dim', type=int, default=64,
                        help='the size to project x to before calculating att scores')
    parser.add_argument("--heads", type=int, default=4, help="number of attention heads")
    parser.add_argument("--attention_norm_idx", type=int, default=0, help="0 = normalise rows, 1 = normalise cols")
    parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                        help='apply a feature transformation xW to the ODE')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, SDE')
    parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT, SDE')
    parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                        help="multiply attention scores by edge weights before softmax")
    # ray args
    parser.add_argument("--num_samples", type=int, default=200, help="number of ray trials")
    parser.add_argument("--gpus", type=float, default=1.0, help="number of gpus per trial. Can be fractional")
    parser.add_argument("--cpus", type=float, default=1, help="number of cpus per trial. Can be fractional")
    parser.add_argument(
        "--grace_period", type=int, default=5, help="number of epochs to wait before terminating trials"
    )
    parser.add_argument(
        "--reduction_factor", type=int, default=4, help="number of trials is halved after this many epochs"
    )
    parser.add_argument("--name", type=str, default="ray_exp")
    parser.add_argument("--num_splits", type=int, default=0, help="Number of random splits >= 0. 0 for planetoid split")
    parser.add_argument("--num_init", type=int, default=1, help="Number of random initializations >= 0")

    parser.add_argument("--max_nfe", type=int, default=300, help="Maximum number of function evaluations allowed in an epoch.")
    parser.add_argument('--metric', type=str, default='accuracy',
                        help='metric to sort the hyperparameter tuning runs on')
    # regularisation args
    parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

    parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

    parser.add_argument("--baseline", action="store_true", help="Wheather to run the ICML baseline or not.")
    parser.add_argument("--regularise", dest='regularise', action='store_true', help='search over reg params')

    # rewiring args
    parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
    parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
    parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
    parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
    parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                        help="above this edge weight, keep edges when using threshold")
    parser.add_argument('--gdc_avg_degree', type=int, default=64,
                        help="if gdc_threshold is not given can be calculated by specifying avg degree")
    parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
    parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
    parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
    parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                        help='incorporate the feature grad in attention based edge dropout')
    parser.add_argument("--exact", action="store_true",
                        help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
    parser.add_argument('--att_samp_pct', type=float, default=1,
                        help="float in [0,1). The percentage of edges to retain based on attention scores")


    # data args
    parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true',
                        help='use the 10 fixed splits from '
                            'https://arxiv.org/abs/2002.05287')
    parser.add_argument('--planetoid_split', action='store_true',
                        help='use planetoid splits for Cora/Citeseer/Pubmed')
    
    
    # Attention args
    parser.add_argument('--attention_type', type=str, default="cosine_sim",
                        help="scaled_dot,cosine_sim,pearson, exp_kernel")
    parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

    # rewiring args

    # beltrami args
    parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
    parser.add_argument('--fa_layer', action='store_true', help='add a bottleneck paper style layer with more edges')
    parser.add_argument('--pos_enc_type', type=str, default="DW64",
                        help='positional encoder either GDC, DW64, DW128, DW256')
    parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
    parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
    parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
    parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
    parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
    parser.add_argument('--edge_sampling_add_type', type=str, default="importance",
                        help="random, ,anchored, importance, degree")
    parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
    parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--edge_sampling_space', type=str, default="attention",
                        help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
    parser.add_argument('--symmetric_attention', action='store_true',
                        help='maks the attention symmetric for rewring in QK space')

    parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
    parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")

    parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")


    args = parser.parse_args()

    opt = vars(args)

    main(opt)
