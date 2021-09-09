import json
from pathlib import Path

import darwin
from darwin.dataset.utils import make_class_lists
from darwin.torch import get_dataset

dataset_id = "cotreat/arch-photos"
# release_path = Path.home() / ".darwin" / "datasets" / "cotreat" / "arch-photos" / "releases" / "support"

# api_key = "B-Nf29s.sb_vKt9jZe52LPczlvCYM8rv4y-Cof65"
# client = darwin.Client.from_api_key(api_key)
# dataset = client.get_remote_dataset("cotreat/arch-photos")
# dataset.pull(video_frames=True)
# dataset.split_video_annotations()

# make_class_lists(release_path)

dataset = get_dataset(dataset_id, dataset_type="instance-segmentation")

for example in dataset:
    # assert example[0].size == (example[1]["width"], example[1]["height"])
    # assert example[0].size == tuple(example[1]["masks"].size())[::-1][:2]

    p = example[1]["image_path"]

    elems = p.split("/")
    name = Path("/".join(elems[:7])) / "releases" / "latest" / "annotations" / f"{Path(elems[-1]).stem}.json"

    with open(name, "r") as f:
        j = json.load(f)

    if example[0].size == (j["image"]["width"], j["image"]["height"]):
        continue

    print(f"{p}'s annotation has wrong sizes!")
