"""Inference Entrypoint script."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from flask import Flask, render_template, request
app = Flask(__name__)

def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a config file")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--input", type=Path, required=True, help="Path to image(s) to infer.")
    parser.add_argument("--output", type=str, required=False, help="Path to save the output image(s).")
    parser.add_argument(
        "--visualization_mode",
        type=str,
        required=False,
        default="simple",
        help="Visualization mode.",
        choices=["full", "simple"],
    )
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )

    return parser

@app.route('/detect-padim', methods=['POST'])
def infer():
    image_path = request.form.get('image_path')

    # print("ok bro", request.form.get('image_path'))

    # print('-------------------Inside Infer function ----------------')
    """Run inference."""
    config = get_configurable_parameters(config_path="./padim_new.yaml")
    config.trainer.resume_from_checkpoint = str("./model-v1.ckpt")
    # config.visualization.show_images = args.show
    # config.visualization.mode = args.visualization_mode
    if "./resultImg":  # overwrite save path
        config.visualization.save_images = True
        config.visualization.image_save_path = "./resultImg"
    else:
        config.visualization.save_images = False

    # print('-------------------Creating model and trainer ----------------')

    # create model and trainer
    model = get_model(config)
    callbacks = get_callbacks(config)
    trainer = Trainer(callbacks=callbacks, **config.trainer)

    # get the transforms
    transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
    image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = tuple(center_crop)
    normalization = InputNormalizationMethod(config.dataset.normalization)
    transform = get_transforms(
        config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization
    )
    # print('-------------------Creating dataset ----------------')

    # create the dataset
    dataset = InferenceDataset(
        image_path, image_size=tuple(config.dataset.image_size), transform=transform  # type: ignore
    )
    dataloader = DataLoader(dataset)

    # generate predictions
    trainer.predict(model=model, dataloaders=[dataloader])
    # sys.stdout = sys.__stdout__  # Restore stdout
    # result = trainer.predict(model=model, dataloaders=[dataloader])
    # print(result)
    # try:
    #     # generate predictions
    #     trainer.predict(model=model, dataloaders=[dataloader])
    #     return True
    # except Exception as e:
    #     print(f"Error occurred during inference: {e}")
    #     return False

    return {"success": True}


if __name__ == "__main__":
    # print('-------------------Before Arg Parser function ----------------')
    # args = get_parser().parse_args()
    # print('-------------------Printing Args ----------------')
    # print(args, flush=True)
    # print('-------------------Before infer function ----------------')
    # infer(args)

    app.run(debug=True, host='localhost', port=5012)
    # app.run(debug=True, host="0.0.0.0")
