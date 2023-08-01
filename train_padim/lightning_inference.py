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

@app.route('/detect-padim', methods=['POST'])
def infer():
    image_path = request.form.get('image_path')

    config = get_configurable_parameters(config_path="./padim_new.yaml")
    config.trainer.resume_from_checkpoint = str("./model-v1.ckpt")

    if "./resultImg":
        config.visualization.save_images = True
        config.visualization.image_save_path = "./resultImg"
    else:
        config.visualization.save_images = False

    model = get_model(config)
    callbacks = get_callbacks(config)
    trainer = Trainer(callbacks=callbacks, **config.trainer)

    transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
    image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = tuple(center_crop)
    normalization = InputNormalizationMethod(config.dataset.normalization)
    transform = get_transforms(
        config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization
    )

    dataset = InferenceDataset(
        image_path, image_size=tuple(config.dataset.image_size), transform=transform
    )
    dataloader = DataLoader(dataset)

    trainer.predict(model=model, dataloaders=[dataloader])

    return {"success": True}

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=5012)
