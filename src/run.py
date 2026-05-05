import os
from os.path import dirname, abspath
import pprint
import shutil
import time
import datetime
import threading
from types import SimpleNamespace as SN

import torch as th
import numpy as np

from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.general_reward_support import test_alg_config_supports_reward
from utils.logging import Logger
from utils.timehelper import time_left, time_str

from functools import partial
import copy
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import utils.hp as hp


def _objective(trial, args_dict, _log):
    param = copy.deepcopy(args_dict)
    if "qmix" in param["name"]:
        param = hp.hp_qmix_settings(trial, param)
    elif "ltscg" in param["name"]:
        param = hp.hp_ltscg_settings(trial, param)
    elif "dicg" in param["name"]:
        param = hp.hp_dicg_settings(trial, param)
    else:
        param = hp.hp_mappo_settings(trial, param)
    match param["agent"]:
        case "mlp":
            param = hp.hp_mlp_settings(trial, param)
        case "rnn":
            param = hp.hp_rnn_settings(trial, param)
        case "gnn":
            param = hp.hp_gnn_settings(trial, param)
        case "gnn_rnn":
            param = hp.hp_gnn_rnn_settings(trial, param)
        case "rnn_gnn":  # same as gnn_rnn
            param = hp.hp_gnn_rnn_settings(trial, param)
        case "tgn":
            param = hp.hp_tgn_settings(trial, param)
        case "egcn":
            param = hp.hp_egcn_settings(trial, param)
    param["t_max"] = int(param["t_max"] / 2)  # we only tune for fast learning
    param["save_model"] = False  # no need to save
    param["trial"] = trial  # for trial.prunning
    print("selected param: ", param, "\n---\n")
    hp_args = SN(**param)
    hp_logger = Logger(_log)
    try:
        run_sequential(args=hp_args, logger=hp_logger)
    except Exception as e:
        hp_logger.console_logger.exception(
            f"error handle, this setting is not good... {str(e)}"
        )
        raise optuna.TrialPruned()
    # we return the 25% last time_step mean of the return mean curve
    start = int(len(hp_logger.stats["test_return_mean"]) * 0.25)
    tmp_res = np.round(
        np.mean([x[1].item() for x in hp_logger.stats["test_return_mean"][-start:]]), 2
    )
    return tmp_res


def _run_optim(args_dict, _log):
    args_dict["seed"] = 42
    args_dict = init_seed(args_dict)
    sampler = optuna.samplers.TPESampler(
        multivariate=True, warn_independent_sampling=False, seed=42
    )
    pruner = optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=5)
    study = optuna.create_study(
        study_name=f"{args_dict['hp_search']} search for {args_dict['unique_token']}",
        storage=JournalStorage(JournalFileBackend(file_path="./journal.log")),
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
    )
    obj = partial(_objective, args_dict=args_dict, _log=_log)
    study.optimize(
        obj, n_trials=args_dict["hp_search"], n_jobs=1, show_progress_bar=True
    )
    return study


def init_seed(config):
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config["env_args"]["seed"] = config["seed"]
    return config


def run(_run, _config, _log):
    _config = init_seed(_config)
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    assert test_alg_config_supports_reward(args), (
        "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."
    )

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"  # _{datetime.datetime.now()}

    args.unique_token = unique_token
    args_dict = vars(args)
    if args.hp_search != 0:
        study = _run_optim(args_dict=args_dict, _log=_log)
        args_dict = hp.update_hp(study, args_dict, map_name, args.name)
        args = SN(**args_dict)

    # setup loggers
    logger = Logger(_log)
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    if args.use_wandb:
        logger.setup_wandb(
            _config, args.wandb_team, args.wandb_project, args.wandb_mode
        )

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Finish logging
    logger.finish()

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    # For individual rewards in gymmai reward is of shape (1, n_agents)
    if args.common_reward:
        scheme["reward"] = {"vshape": (1,)}
    else:
        scheme["reward"] = {"vshape": (args.n_agents,)}
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            batch = episode_sample[:, :max_ep_t]

            if batch.device != args.device:
                batch.to(args.device)

            if args.name == "ltscg":
                learner.train(episode_sample, runner.t_env, episode)
            else:
                learner.train(batch, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

            if args.use_wandb and args.wandb_save_model:
                wandb_save_dir = os.path.join(
                    logger.wandb.dir, "models", args.unique_token, str(runner.t_env)
                )
                os.makedirs(wandb_save_dir, exist_ok=True)
                for f in os.listdir(save_path):
                    shutil.copyfile(
                        os.path.join(save_path, f), os.path.join(wandb_save_dir, f)
                    )

        episode += args.batch_size

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
            if hasattr(args, "trial"):
                tmp_res = np.round(logger.stats["test_return_mean"][-1][1].item(), 2)
                args.trial.report(tmp_res, runner.t_env)
                # Handle pruning based on the intermediate value.
                if args.trial.should_prune() and runner.t_env >= int(
                    args.t_max / 2
                ):  # prune disable before 1/2 of t_max (1/4 of t_max*2)
                    runner.close_env()
                    raise optuna.TrialPruned()

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size"]:
        config["test_nepisode"] = config["batch_size"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size"]
        ) * config["batch_size"]

    return config
