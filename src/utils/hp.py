import yaml
from optuna.trial import TrialState
from optuna import Study, Trial, visualization
from plotly.io import write_image

# This reference give some advices : https://arxiv.org/abs/2306.01324

def update_hp(study: Study, hp: dict, tuned_path: str) -> dict:
    """
    load and save best parameters found during the search

    Args:
        study (Study):
        hp (dict):
        tuned_path (str):

    Returns:
        dict:
    """
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial: ")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        hp[key] = value

    if hp["runner"] == "parallel":  # set the buffer size as the same than the batch size
        hp["buffer_size"] = hp["batch_size"]

    # os.makedirs(os.path.dirname(tuned_path) or ".", exist_ok=True)

    with open(tuned_path, "w", encoding="utf-8") as fichier:
        yaml.dump(hp, fichier, indent=4)

    fig_importance = visualization.plot_param_importances(study)
    fig_history = visualization.plot_optimization_history(study)
    fig_relation = visualization.plot_parallel_coordinate(study)
    write_image(fig_importance, f"{tuned_path}_hp_importances.png")
    write_image(fig_history, f"{tuned_path}_hp_history.png")
    write_image(fig_relation, f"{tuned_path}_hp_relation.png")
    return hp


def hp_mappo_settings(trial: Trial, hp: dict) -> dict:
    """
    suggest params for classic mappo train process

    Args:
        trial (Trial): 
        hp (dict): 

    Returns:
        dict: updated params
    """
    hp["lr"] = trial.suggest_float("lr", 1e-6, 0.1, log=True)
    hp["eps_clip"] = trial.suggest_float("eps_clip", 0.01, 0.5)

    hp["q_nstep"] = trial.suggest_int("q_nstep", 5, 20)
    hp["entropy_coef"] = trial.suggest_float("entropy_coef", 0.0, 0.5)

    hp["grad_norm_clip"] = trial.suggest_int("grad_norm_clip", 1, 20)
    hp["standardise_returns"] = trial.suggest_categorical("standardise_returns", [False, True])
    hp["target_update_interval_or_tau"] = trial.suggest_float("target_update_interval_or_tau", 0.01, 1.0)

    hp["epochs"] = trial.suggest_int("epochs", 5, 20)
    hp["batch_size"] = trial.suggest_int("batch_size", 16, 128, step=16)
    hp["buffer_size"] = hp["batch_size"]

    return hp

def hp_mlp_settings(trial: Trial, hp: dict) -> dict:
    """
    suggest params for mlp architecture

    Args:
        trial (Trial): 
        hp (dict): 

    Returns:
        dict: updated params
    """
    hp["n_layers"] = trial.suggest_int("n_layers", 1, 3)
    hp["h_dim"] = trial.suggest_int("h_dim", 64, 512, step=64)
    hp["layer_norm"] = trial.suggest_categorical("layer_norm", [False, True])

    return hp

def hp_rnn_settings(trial: Trial, hp: dict) -> dict:
    """
    suggest params for rnn architecture

    Args:
        trial (Trial): 
        hp (dict): 

    Returns:
        dict: updated params
    """
    hp = hp_mlp_settings(trial, hp)
    hp["mem_dim"] = trial.suggest_int("mem_dim", 64, 512, step=64)
    return hp

def hp_gnn_settings(trial: Trial, hp: dict) -> dict:
    """
    suggest params for gnn architecture

    Args:
        trial (Trial): 
        hp (dict): 

    Returns:
        dict: updated params
    """
    hp = hp_mlp_settings(trial, hp)
    hp["gnn_dim"] = trial.suggest_int("gnn_dim", 64, 512, step=64)
    hp["n_heads_gat"] = trial.suggest_int("n_heads_gat", 1, 3)
    hp["dropout_gat"] = trial.suggest_categorical("dropout_gat", [0, 0.25, 0.5])
    hp["residual_gat"] = trial.suggest_categorical("residual_gat", [False, True])
    hp["edge_attr"] = trial.suggest_categorical("edge_attr", [False, True])
    return hp
