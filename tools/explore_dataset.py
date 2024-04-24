import sys
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch

from mmdet3d.datasets import build_dataset
from accelerate.utils import set_seed

# fmt: off
# bypass annoying warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
# fmt: on

sys.path.append(".")  # noqa
from magicdrive.dataset import *


def collate_fn(examples):
    return examples


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # setup logger
    # only log debug info to log file
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
    # handle log from some packages
    logging.getLogger("shapely.geos").setLevel(logging.WARN)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("accelerate.tracking").setLevel(logging.INFO)
    logging.getLogger("numba").setLevel(logging.WARN)
    logging.getLogger("PIL").setLevel(logging.WARN)
    logging.getLogger("matplotlib").setLevel(logging.WARN)
    setattr(cfg, "log_root", HydraConfig.get().runtime.output_dir)

    # since our model has randomness to train the uncond embedding, we need this.
    set_seed(cfg.seed)

    # datasets
    train_dataset = build_dataset(
        OmegaConf.to_container(cfg.dataset.data.train, resolve=True)
    )
    val_dataset = build_dataset(
        OmegaConf.to_container(cfg.dataset.data.val, resolve=True)
    )

    for dataset in [train_dataset, val_dataset]:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=cfg.runner.train_batch_size,
            num_workers=cfg.runner.num_workers,
            pin_memory=True,
            prefetch_factor=cfg.runner.prefetch_factor,
            persistent_workers=True,
        )

        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = 0, 0, 0
        max_boxes = 0
        max_desc_len = 0
        num = 0
        logging.info("Run with new dataset")
        for data in tqdm(dataloader):
            # tokens = [di['metas'].data['token'] for di in data]
            # if "a7e83e3b5cd24948a58c93c66ba54d77" in tokens:
            #     logging.info(f"num={num}, tokens={tokens}")
            #     exit(0)
            # else:
            #     num += len(tokens)
            #     continue
            boxes = []
            for datai in data:
                if len(datai['gt_bboxes_3d'].data.tensor) != 0:
                    boxes.append(datai['gt_bboxes_3d'].data.corners)
                    max_boxes = max(
                        max_boxes, len(datai['gt_bboxes_3d'].data.corners))
                    max_desc_len = max(
                        max_desc_len,
                        datai['metas'].data['description'].split(" ").__len__(),
                    )
                else:
                    logging.info(f"no box on {datai['metas'].data['token']}")
            if len(boxes) == 0:
                continue
            boxes = torch.cat(boxes, dim=0)

            max_x = max(max_x, boxes[:, :, 0].max())
            min_x = min(min_x, boxes[:, :, 0].min())

            max_y = max(max_y, boxes[:, :, 1].max())
            min_y = min(min_y, boxes[:, :, 1].min())

            max_z = max(max_z, boxes[:, :, 2].max())
            min_z = min(min_z, boxes[:, :, 2].min())
        logging.info(f"max x, y, z = {max_x}, {max_y}, {max_z}")
        logging.info(f"min x, y, z = {min_x}, {min_y}, {min_z}")
        logging.info(f"max boxes = {max_boxes}")
        logging.info(f"max description length = {max_desc_len}")


if __name__ == "__main__":
    main()
