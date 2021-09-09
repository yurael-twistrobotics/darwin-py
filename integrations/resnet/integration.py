from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from darwin.datatypes import make_tag
from darwin.integrations import IntegrationStrategy
from darwin.integrations.train_handler import TrainHandler
from darwin.integrations.typing import (
    InferParams,
    InferResult,
    LoadParams,
    TrainParams,
    TrainResult,
)
from PIL import Image


class Integration(IntegrationStrategy):
    def load(self, weights_path: Optional[Path], params: LoadParams) -> None:
        self.classes = params.classes

        pretrained = weights_path is None
        self.model = models.resnet50(pretrained=pretrained)

        self.model.features = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.classifier = torch.nn.Sequential(torch.nn.Linear(512, len(self.classes)))

        if weights_path:
            self.model.load_state_dict(torch.load(weights_path))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def infer(self, image_paths: List[Path], params: InferParams) -> InferResult:
        transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.CenterCrop((224, 224)), transforms.ToTensor()]
        )

        images = [transform(Image.open(image_path)) for image_path in image_paths]
        image_batch = torch.stack(images).to(self.device)

        self.model.eval()
        output = self.model(image_batch)
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1).item()

        tag = make_tag(self.classes[output]["name"])
        return [tag]

    def train(self, train_handler: TrainHandler, params: TrainParams) -> TrainResult:
        raise NotImplementedError()


if __name__ == "__main__":
    classes = Path("classes.txt").read_text().split("\n")

    integration = Integration()
    integration.load(None, {"classes": classes})
    print(integration.infer([Path("500.jpg")], {}))
