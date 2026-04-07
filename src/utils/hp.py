import yaml
from optuna.trial import TrialState
from optuna import Study, Trial, visualization
from plotly.io import write_image


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
    hp["grad_norm_clip"] = trial.suggest_int("grad_norm_clip", 5, 20, step=5)
    hp["lr"] = trial.suggest_float("lr", 0.0001, 0.001, log=True)
    hp["entropy_coef"] = trial.suggest_categorical("entropy_coef", [0.0, 0.001, 0.01, 0.1])
    hp["add_value_last_step"] = trial.suggest_categorical("add_value_last_step", [True, False])
    hp["standardise_returns"] = trial.suggest_categorical("standardise_returns", [True, False])
    hp["tau"] = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.05, 0.1])

    hp["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    hp["buffer_size"] = hp["batch_size"]

    hp["q_nstep"] = trial.suggest_categorical("q_nstep", [1, 5, 10, 15, 20])
    hp["epochs"] = trial.suggest_categorical("epochs", [5, 10, 15, 20])
    hp["eps_clip"] = trial.suggest_float("eps_clip", 0.01, 0.3, log=True)

    hp["obs_agent_id"] = trial.suggest_categorical("obs_agent_id", [True, False])
    hp["obs_last_action"] = trial.suggest_categorical("obs_last_action", [True, False])

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
    hp["hidden_dim"] = trial.suggest_int("hidden_dim", 64, 512, step=64)
    hp["layer_norm"] = trial.suggest_categorical("layer_norm", [True, False])

    return hp
