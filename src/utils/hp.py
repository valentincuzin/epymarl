import yaml
import os
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
    hp["eps_clip"] = trial.suggest_float("eps_clip", 0.0, 0.5)

    hp["q_nstep"] = trial.suggest_int("q_nstep", 5, 20, step=5)
    hp["entropy_coef"] = trial.suggest_float("entropy_coef", 0.0, 0.5)

    hp["grad_norm_clip"] = trial.suggest_int("grad_norm_clip", 5, 20, step=5)
    hp["add_value_last_step"] = trial.suggest_categorical("add_value_last_step", [True, False])
    hp["standardise_returns"] = trial.suggest_categorical("standardise_returns", [True, False])
    hp["target_update_interval_or_tau"] = trial.suggest_categorical("target_update_interval_or_tau", [0.01, 0.05, 0.1])

    hp["epochs"] = trial.suggest_int("epochs", 5, 20, step=5)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
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
    hp["hidden_dim"] = trial.suggest_int("hidden_dim", 64, 512, step=64)
    hp["layer_norm"] = trial.suggest_categorical("layer_norm", [True, False])

    return hp
