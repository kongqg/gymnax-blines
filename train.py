import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import jax
from utils.models import get_model_ready
from utils.helpers import load_config, save_pkl_object
import wandb



def main(config, mle_log, log_ext=""):
    wandb.init(
        project="gymnax-blines-kongqg",
        entity="kongqg574-1",
        config=config,
        name=f"{config.env_name}_{config.algo}_{config.tau}_seed{config.seed_id}",
        group=config.env_name,                 #
        job_type=f"tau_{config.tau}",          #
        tags=[config.algo, f"tau={config.tau}"]# 
    )



    """Run training with ES or PPO. Store logs and agent ckpt."""
    rng = jax.random.PRNGKey(config.seed_id)
    # Setup the model architecture
    rng, rng_init = jax.random.split(rng)
    model, params = get_model_ready(rng_init, config)

    # Run the training loop (either evosax ES or PPO)
    if config.train_type == "ES":
        from utils.es import train_es as train_fn
    elif config.train_type == "PPO":
        from utils.ppo_algos import train_ppo as train_fn
    else:
        raise ValueError("Unknown train_type. Has to be in ('ES', 'PPO').")

    # Log and store the results.
    log_steps, log_return, network_ckpt = train_fn(
        rng, config, model, params, mle_log
    )

    data_to_store = {
        "log_steps": log_steps,
        "log_return": log_return,
        "network": network_ckpt,
        "train_config": config,
    }

    save_pkl_object(
        data_to_store,
        f"agents/{config.env_name}/{config.train_type.lower()}{log_ext}.pkl",
    )
    wandb.finish()

if __name__ == "__main__":
    # Use MLE-Infrastructure if available (e.g. for parameter search)
    # try:
    #     from mle_toolbox import MLExperiment

    #     mle = MLExperiment(config_fname="configs/cartpole/ppo.yaml")
    #     main(mle.train_config, mle_log=mle.log)
    # # Otherwise use simple logging and config loading
    # except Exception:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        "--config_fname",
        type=str,
        default="configs/CartPole-v1/ppo.yaml",
        help="Path to configuration yaml.",
    )
    parser.add_argument(
        "-seed",
        "--seed_id",
        type=int,
        default=0,
        help="Random seed of experiment.",
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "dhvl"],
        help="Algorithm to run.",
)
    parser.add_argument(
        "-lr",
        "--lrate",
        type=float,
        default=5e-04,
        help="Random seed of experiment.",
    )

    parser.add_argument(
    "--tau",
    type=float,
    default=None,   # 不传就用 yaml 里的 tau
    help="Tau parameter (override config train_config.tau).",
)
    parser.add_argument(
        "--value_update_iter",
        type=int,
        default=None,   # 不传就用 yaml 里的 tau
        help="value_update_iter",
    )
    parser.add_argument(
        "--actor_update_iter",
        type=int,
        default=None,   # 不传就用 yaml 里的 tau
        help="actor_update_iter",
    )
    args, _ = parser.parse_known_args()
    config = load_config(args.config_fname, args.seed_id, args.lrate)

    config.train_config.algo = args.algo
    config.train_config.value_update_iter = args.value_update_iter
    config.train_config.actor_update_iter = args.actor_update_iter

    if args.tau is not None:
        config.train_config.tau = float(args.tau)
    main(
        config.train_config,
        mle_log=None,
        log_ext=str(args.lrate) if args.lrate != 5e-04 else "",
    )
