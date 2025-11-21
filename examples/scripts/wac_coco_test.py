import yaml
from terratorch.models import ObjectDetectionModelFactory
from terratorch.datamodules.wac_robbins import WACVisRobbinsDataModule
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from terratorch.tasks import ObjectDetectionTask
import torch
import pdb
import json

CALLBACK_REGISTRY = {
    "RichProgressBar": RichProgressBar,
    "LearningRateMonitor": LearningRateMonitor,
    "ModelCheckpoint": ModelCheckpoint,
    "EarlyStopping": EarlyStopping,
}

def instantiate_callbacks(config):
    instances = []

    for cb_cfg in config["callbacks"]:
        class_path = cb_cfg["class_path"]
        init_args = cb_cfg.get("init_args", {})

        cls = CALLBACK_REGISTRY[class_path]
        cb = cls(**init_args)
        instances.append(cb)

    return instances

with open("/dccstor/cimf/gabby/object_detection/terratorch_configs/wac_robbins_timm_convnextv2.yaml","r") as f:
    config = yaml.safe_load(f)
data_init_args = config['data']['init_args']
wac_vis_datamodule = WACVisRobbinsDataModule( wac_data_root=data_init_args['wac_data_root'], 
                                            #  coco_data_root=data_init_args['coco_data_root'],
                                            stats_path=data_init_args['stats_path'],
                                            splits_path=data_init_args['splits_path'],
                                            annotations_path=data_init_args['annotations_path'],
                                            num_workers = data_init_args['num_workers'],
                                            batch_size = data_init_args['batch_size'],
                                            # This is the default - use 415 as the RBG bands
                                            bands= ('415','415','415'),
                                            no_data_replace= 0,
                                            # This applies to the percentile_normalization in my dataset
                                            apply_norm_in_datamodule=data_init_args['apply_norm_in_datamodule'],
                                            percentile_normalize=data_init_args['percentile_normalize']
                                            )
wac_vis_datamodule.setup(stage="fit")
# pdb.set_trace()
s = wac_vis_datamodule.train_dataset[0]
print("unique labels:", torch.unique(s["labels"]))

model_init_args = config['model']['init_args']
# model_args = dict(model_init_args["model_args"]) 
task = ObjectDetectionTask(
    model_args = model_init_args['model_args'],
    model_factory = "ObjectDetectionModelFactory",
    lr=config['optimizer']['init_args']['lr'],
    optimizer=config['optimizer']['class_path'].replace("torch.optim.",""),
    optimizer_hparams=config['optimizer']['init_args'],
    scheduler = config['lr_scheduler']['class_path'],
    scheduler_hparams = config['lr_scheduler']['init_args'],
    freeze_backbone = model_init_args['freeze_backbone'],
    freeze_decoder = model_init_args['freeze_decoder'],
    class_names = model_init_args['class_names'],
    # TEST TO SHOW MORE PREDICTIONS (before score threshold was set to 0.5)
    # score_threshold=0.3,
    # iou_threshold=0.5,
)

callbacks = instantiate_callbacks(config["trainer"])


logger = TensorBoardLogger(
    save_dir = config['trainer']['logger']['init_args']['save_dir'],
    name = config['trainer']['logger']['init_args']['name'])

config_trainer = config['trainer']
trainer = pl.Trainer(
    accelerator=config_trainer['accelerator'],
    strategy=config_trainer['strategy'],
    devices=config_trainer['devices'],
    num_nodes=config_trainer['num_nodes'],
    precision=config_trainer['precision'],
    max_epochs=config_trainer['max_epochs'],
    check_val_every_n_epoch=config_trainer['check_val_every_n_epoch'],
    log_every_n_steps=config_trainer['log_every_n_steps'],
    enable_checkpointing=config_trainer['enable_checkpointing'],
    default_root_dir=config_trainer['default_root_dir'],
    callbacks = callbacks,
    logger=logger
)

_ = trainer.fit(model=task, datamodule=wac_vis_datamodule)

wac_vis_datamodule.setup(stage="test")
res = trainer.test(model=task, datamodule=wac_vis_datamodule)
print(res)

with open(f"{config_trainer['default_root_dir']}/test_metrics.json", "w") as f:
    json.dump(res, f)