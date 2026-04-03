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

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        hp[key] = value

    with open(f"{tuned_path}.json", "w", encoding="utf-8") as fichier:
        yaml.dump(hp, fichier, ensure_ascii=False, indent=4)

    fig_importance = visualization.plot_param_importances(study)
    fig_history = visualization.plot_optimization_history(study)
    fig_relation = visualization.plot_parallel_coordinate(study)
    write_image(fig_importance, f"{tuned_path}_hp_importances.png")
    write_image(fig_history, f"{tuned_path}_hp_history.png")
    write_image(fig_relation, f"{tuned_path}_hp_relation.png")
    return hp
