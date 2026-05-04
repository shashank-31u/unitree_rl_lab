from dataclasses import field
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class CleanMLPModelCfg:
    """Minimal MLP model config with only fields accepted by rsl-rl-lib 5.x MLPModel."""

    class_name: str = "MLPModel"
    hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    obs_normalization: bool = False
    distribution_cfg: dict | None = None


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""
    empirical_normalization = False

    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    actor = CleanMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    critic = CleanMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
