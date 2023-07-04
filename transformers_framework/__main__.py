import os

import datasets
import lightning.pytorch as pl
import torch
import torch._dynamo
import torchmetrics
import transformers
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger

from transformers_framework import __version__
from transformers_framework.callbacks.early_stopping import EarlyStopping
from transformers_framework.callbacks.prediction_writer import PredictionsWriter
from transformers_framework.callbacks.rich_progress_bar import RichProgressBar
from transformers_framework.callbacks.transformers_checkpoint import TransformersModelCheckpointCallback
from transformers_framework.datamodules.arrow_datamodule import ArrowDataModule
from transformers_framework.pipelines import pipelines
from transformers_framework.utilities.arguments import (
    FlexibleArgumentParser,
    add_trainer_args,
    get_trainer_args_from_hyperparameters,
)
from transformers_framework.utilities.classes import ExtendedNamespace
from transformers_framework.utilities.logging import rank_zero_info, rank_zero_warn


# too much complains of the tokenizers
transformers.logging.set_verbosity_error()
# when using workers in dataloaders it is better to disable tokenizers parallelism
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# reduce memory usage
datasets.config.IN_MEMORY_MAX_SIZE = 0

TENSORBOARD_DIR = 'tensorboard'
CHECKPOINTS_DIR = 'checkpoints'
PRE_TRAINED_DIR = "pre_trained_models"
PREDICTIONS_DIR = "predictions"

# set verbose mode with Dynamo for better error traces when compiling
torch._dynamo.config.verbose = True


def main(hyperparameters: ExtendedNamespace):

    # Print systems info
    rank_zero_info(
        f"Starting experiment '{hyperparameters.name}', with model "
        f"'{hyperparameters.model}' and pipeline '{hyperparameters.pipeline}'..."
    )
    rank_zero_info(
        f"Running on\n"
        f"  - transformers_framework={__version__}\n"
        f"  - torch={torch.__version__}\n"
        f"  - transformers={transformers.__version__}\n"
        f"  - pytorch-lightning={pl.__version__}\n"
        f"  - datasets={datasets.__version__}\n"
        f"  - torchmetrics={torchmetrics.__version__}\n"
    )

    # compute relative paths. tensorboard will add name by itself
    checkpoints_path = os.path.join(hyperparameters.output_dir, CHECKPOINTS_DIR, hyperparameters.name)
    tensorboard_path = os.path.join(hyperparameters.output_dir, TENSORBOARD_DIR)
    pre_trained_path = os.path.join(hyperparameters.output_dir, PRE_TRAINED_DIR, hyperparameters.name)
    predictions_path = os.path.join(hyperparameters.output_dir, PREDICTIONS_DIR, hyperparameters.name)

    # enable fallback to eager execution to avoid errors
    if hyperparameters.dynamo_fallback_eager:
        rank_zero_warn("Suppressing Dynamo errors in torch to avoid hang training.")
        torch._dynamo.config.suppress_errors = True

    # set the random seed
    seed_everything(seed=hyperparameters.seed, workers=True)

    # instantiate PL model
    pl_model_class = pipelines[hyperparameters.pipeline][hyperparameters.model]
    model = pl_model_class(hyperparameters)

    # trainer additional arguments
    kwargs = dict(default_root_dir=hyperparameters.output_dir)

    # default jsonboard logger
    logger = TensorBoardLogger(save_dir=tensorboard_path, name=hyperparameters.name)
    kwargs['logger'] = logger

    # save pre-trained models to
    save_transformers_callback = TransformersModelCheckpointCallback(hyperparameters, destination=pre_trained_path)

    # log learning rate
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    # save PL checkpoints
    checkpoint_callback_args = dict(verbose=True, dirpath=checkpoints_path, save_weights_only=False)

    if hyperparameters.monitor is not None:
        checkpoint_callback_args = dict(
            **checkpoint_callback_args,
            monitor=hyperparameters.monitor,
            save_last=True,
            mode=hyperparameters.monitor_direction,
            save_top_k=1,
            every_n_train_steps=hyperparameters.checkpoint_interval,
        )
    checkpoint_callback = ModelCheckpoint(**checkpoint_callback_args)

    # rich progress bar
    rich_progress_bar = RichProgressBar(leave=True)

    # modelsummary callback
    model_summary = RichModelSummary(max_depth=2)

    # all callbacks
    callbacks = [
        save_transformers_callback,
        lr_monitor_callback,
        checkpoint_callback,
        rich_progress_bar,
        model_summary,
    ]

    # early stopping if defined
    if hyperparameters.early_stopping:
        early_stopping_callback = EarlyStopping(hyperparameters)
        callbacks.append(early_stopping_callback)

    # add Stochastic Weight Averaging
    if hyperparameters.stochastic_weight_averaging:
        callbacks.append(StochasticWeightAveraging(swa_lrs=hyperparameters.learning_rate * 100, swa_epoch_start=0.75))

    # add callback for predictions
    callbacks.append(PredictionsWriter(hyperparameters, destination=predictions_path))

    kwargs['callbacks'] = callbacks

    # instantiate PL trainer
    trainer = pl.Trainer(**get_trainer_args_from_hyperparameters(hyperparameters), **kwargs)

    # DataModules
    datamodule = ArrowDataModule(hyperparameters, trainer, model)

    # Train!
    if datamodule.do_train():
        rank_zero_info(f"Training experiment {hyperparameters.name}")
        trainer.fit(model, datamodule=datamodule, ckpt_path=hyperparameters.ckpt_path)

    # Test!
    if datamodule.do_test():
        rank_zero_info(f"Testing experiment {hyperparameters.name}")
        if datamodule.do_train():  # retrieve best model from training
            if hyperparameters.monitor:
                rank_zero_info(
                    f"Going to test on best ckpt chosen over "
                    f"{hyperparameters.monitor}: {checkpoint_callback.best_model_path}"
                )
                return trainer.test(datamodule=datamodule, ckpt_path='best')
            else:
                rank_zero_info("Going to test on last trained ckpt")
                return trainer.test(datamodule=datamodule)
        else:
            rank_zero_info("Going to directly test on loaded ckpt")
            return trainer.test(model, datamodule=datamodule, ckpt_path=hyperparameters.ckpt_path)

    # Predict!
    if datamodule.do_predict():
        rank_zero_info(f"Predicting for experiment {hyperparameters.name}")
        if not hyperparameters.deterministic:
            raise ValueError("Predict must be run in deterministic mode!")
        trainer.predict(model, datamodule=datamodule, return_predictions=False)

    return 0


if __name__ == '__main__':

    # Read config for defaults and eventually override with hyper-parameters from command line
    parser = FlexibleArgumentParser(
        prog=f"Transformers Framework v{__version__}",
        description="Flexible experiments with Transformers",
        add_help=True,
    )

    # model class name
    parser.add_argument('--pipeline', type=str, required=True, choices=pipelines.keys())

    # retrieving model with temporary parsed arguments
    tmp_params = parser.parse_known_args()[0]
    parser.add_argument('--model', type=str, required=True, choices=pipelines[tmp_params.pipeline].keys())

    # experiment name, used both for checkpointing, pre_trained_names, logging and tensorboard
    parser.add_argument('--name', type=str, required=True, help='Name of the model')

    # various options
    parser.add_argument('--seed', type=int, default=1337, help='Set the random seed')  # nosec

    # validation and early stopping monitors
    parser.add_argument('--early_stopping', action="store_true", help="Use early stopping")
    parser.add_argument('--monitor', type=str, help='Value to monitor for best checkpoint', default=None)
    parser.add_argument(
        '--monitor_direction',
        type=str,
        help='Monitor value direction for best',
        default='max',
        choices=['min', 'max'],
    )

    # Output folder
    parser.add_argument(
        '--output_dir',
        type=str,
        required=False,
        default="outputs",
        help='Specify a different output folder'
    )

    # checkpointing
    parser.add_argument('--ckpt_path', type=str, default=None, help="Restore from checkpoint.", required=False)

    # SWA
    parser.add_argument('--stochastic_weight_averaging', action="store_true", help="Activate SWA")

    # Dynamo suppress errors
    parser.add_argument('--dynamo_fallback_eager', action="store_true", help="Suppress dynamo errors")

    # add all the important trainer options to argparse
    # ie: now --devices --num_nodes ... --fast_dev_run all work in the cli
    add_trainer_args(parser)

    # retrieving model and other args with temporary parsed arguments
    tmp_params = parser.parse_known_args()[0]

    # get pl_model_class in advance to know which params it needs
    pipelines[tmp_params.pipeline][tmp_params.model].add_argparse_args(parser)

    # add datamodule hparams
    ArrowDataModule.add_argparse_args(parser)

    # add callbacks / loggers specific parameters
    if tmp_params.early_stopping:
        EarlyStopping.add_argparse_args(parser)
    TransformersModelCheckpointCallback.add_argparse_args(parser)

    # get NameSpace of parameters
    args = parser.parse_args()
    args = ExtendedNamespace.from_namespace(args)
    main(args)
