best_params_dict = {
    'sc':
    {
        "adaptive": False,
        "add_source": False,
        "adjoint": True,
        "adjoint_method": "dopri5",
        "adjoint_step_size": 1,
        "alpha": 1.0,
        "alpha_dim": "sc",
        "att_samp_pct": 0.75630814018008,
        "attention_dim": 64,
        "attention_norm_idx": 0,
        "attention_type": "scaled_dot",
        "augment": False,
        "avg_degree": 16,
        "baseline": False,
        "batch_norm": True,
        "beltrami": False,
        "beta_dim": "sc",
        "block": "constant",
        "cpus": 8,
        "data_norm": "rw",
        "dataset": "sc",
        "decay": 0.05603105393539114,
        "directional_penalty": 0.0162949577857589,
        "dropout": 0.003262782465475916,
        "dt": 0.001,
        "dt_min": 1e-05,
        "edge_sampling": False,
        "edge_sampling_T": "T0",
        "edge_sampling_add": 0.64,
        "edge_sampling_add_type": "importance",
        "edge_sampling_epoch": 5,
        "edge_sampling_online": False,
        "edge_sampling_online_reps": 4,
        "edge_sampling_rmv": 0.32,
        "edge_sampling_space": "attention",
        "edge_sampling_sym": False,
        "epoch": 100,
        "exact": False,
        "fa_layer": False,
        "fa_layer_edge_sampling_rmv": 0.8,
        "fc_out": False,
        "feat_hidden_dim": 64,
        "function": "transformer",
        "gdc_avg_degree": 64,
        "gdc_k": 32,
        "gdc_method": "ppr",
        "gdc_sparsification": "threshold",
        "gdc_threshold": 0.008620180105731238,
        "geom_gcn_splits": False,
        "gpu": 0,
        "gpus": 1.0,
        "grace_period": 50,
        "heads": 1,
        "heat_time": 3.0,
        "hidden_dim": 128,
        "input_dropout": 0.5,
        "jacobian_norm2": None,
        "kinetic_energy": 0.5382257803303859,
        "label_rate": 1.0,
        "leaky_relu_slope": 0.22917897161989106,
        "lr": 0.0818167221514199,
        "max_iters": 100,
        "max_nfe": 300,
        "method": "dopri5",
        "metric": "accuracy",
        "mix_features": False,
        "name": "sc",
        "no_alpha_sigmoid": False,
        "not_lcc": True,
        "num_init": 1,
        "num_samples": 1,
        "num_splits": 0,
        "ode_blocks": 1,
        "optimizer": "adam",
        "planetoid_split": False,
        "pos_dist_quantile": 0.001,
        "pos_enc_csv": False,
        "pos_enc_hidden_dim": 32,
        "pos_enc_orientation": "row",
        "pos_enc_type": "DW64",
        "ppr_alpha": 0.08645076728634832,
        "reduction_factor": 4,
        "regularise": True,
        "reweight_attention": False,
        "rewiring": "gdc",
        "self_loop_weight": 1,
        "square_plus": False,
        "step_size": 1,
        "symmetric_attention": False,
        "time": 18.88666342591446,
        "tol_scale": 6.325413889386998,
        "tol_scale_adjoint": 1570.6826306261953,
        "total_deriv": None,
        "use_cora_defaults": False,
        "use_flux": False,
        "use_labels": True,
        "use_mlp": False
    },
}
