from pytorch_lightning.trainer.states import RunningStage, TrainerFn


TrainerFn_to_Names = {
    TrainerFn.FITTING: 'train',
    TrainerFn.VALIDATING: 'valid',
    TrainerFn.TESTING: 'test',
    TrainerFn.PREDICTING: 'predict',
}


TrainerStage_to_Names = {
    RunningStage.TRAINING: 'train',
    RunningStage.SANITY_CHECKING: 'valid',
    RunningStage.VALIDATING: 'valid',
    RunningStage.TESTING: 'test',
    RunningStage.PREDICTING: 'predict',
}
