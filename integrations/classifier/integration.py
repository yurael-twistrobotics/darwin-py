from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchmetrics as M
from darwin.integrations import IntegrationStrategy
from darwin.integrations.typing import InferParams, LoadParams
from darwin.integrations.utils import convert_to_rgb
from PIL import Image

from classifier import Classifier
from logger import logger


class Integration(IntegrationStrategy):
    DEFAULT_HYPERPARAMETERS = {"lr": 0.0002, "model_name": "regnety_032", "pretrained": True}

    def load(self, weights_path: Optional[Path], params: LoadParams):
        params.update(self.DEFAULT_HYPERPARAMETERS)
        num_classes = len(params["classes"])
        logger.info(f"[LOAD] weights_path={weights_path} params={params} num_classes={num_classes}")

        # TODO do not download if you have working state_url
        model_name = params["model_name"]
        model = to_finetune(model_name, num_classes)
        criterion = torch.nn.CrossEntropyLoss()

        metrics = M.MetricCollection(
            {
                "accuracy": M.Accuracy(num_classes=num_classes),
                "recall": M.Recall(num_classes=num_classes),
                "f1": M.F1(num_classes=num_classes),
                "precision": M.Precision(num_classes=num_classes),
            }
        )

        classifier = Classifier(model=model, criterion=criterion, metrics=metrics, lr=params["lr"])

        if weights_path is None:
            pretrained = params["pretrained"]
            logger.warning(
                f"[LOAD] No checkpoints given. Creating model_name={model_name} with pretrained={pretrained}"
            )
        else:
            classifier = Classifier.load_from_checkpoint(
                weights_path, model=model, criterion=criterion, metrics=metrics
            )
            logger.info(f"[LOAD] Loaded weights for model_name={params.model_name}")

        self.model = classifier
        self.classes = params["classes"]

    def infer(self, file_paths: Path, params: InferParams):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[INFERENCE] device={device}")

        model = self.model.eval().to(device)
        logger.info(f"[INFERENCE] params={params}")

        with torch.no_grad():
            image = prepare_batch(file_paths)
            image = image.to(device)

            logger.info(f"[INFERENCE] img shape = {image.shape}")
            outs = model(image)
            probs = F.softmax(outs, dim=1)

            logger.info(f"[INFERENCE] probs shape = {probs.shape}")
            preds = torch.argmax(probs, dim=1)
            preds = preds.cpu()
            idx = preds.tolist()[0]
            name = self.classes[idx]["name"]
            confidence = probs.squeeze(0)[idx].item()
            # Use darwin-py data classes
            return [{"name": name, "label": name, "tag": {}, "confidence": confidence}]

    def train(self, train_handler, params):
        pass


def prepare_batch(image_paths: List[Path]) -> torch.Tensor:
    import torchvision.transforms as T

    transform = T.Compose(
        [
            T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
        ]
    )

    logger.info(f"[INFERENCE] proprocessing with transform={transform}")
    image_batch = [transform(convert_to_rgb(Image.open(image_path))) for image_path in image_paths]
    image_batch = torch.stack(image_batch)
    return image_batch


def to_finetune(name: str, num_classes: int) -> torch.nn.Module:
    from glasses.models import AutoModel

    model = AutoModel.from_pretrained(name)
    model = torch.nn.Sequential(
        model.encoder,
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(model.encoder.widths[-1], num_classes),
    )
    return model
