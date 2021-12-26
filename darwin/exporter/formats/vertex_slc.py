import csv
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import darwin.datatypes as dt


def export(annotation_files: Iterator[dt.AnnotationFile], output_dir: Path) -> None:
    train_stems, val_stems, test_stems = _load_random_split(output_dir)

    with open(output_dir.parent / "vertex.csv", "w") as f:
        for annotation_file in annotation_files:
            row = export_file(annotation_file, train_stems, val_stems, test_stems)
            if row is None:
                continue

            writer = csv.writer(f)
            writer.writerow(row)


def export_file(
    annotation_file: dt.AnnotationFile, train_stems: List[str], val_stems: List[str], test_stems: List[str]
) -> Optional[Tuple[str, str, str]]:
    if annotation_file.is_video:
        return None

    tags = [a for a in annotation_file.annotations if a.annotation_class.annotation_type == "tag"]
    if len(tags) != 1:
        return None

    ml_use = _infer_ml_use(annotation_file.filename, train_stems, val_stems, test_stems)
    gs_bucket = "darwin"
    gs_path = f"gs://{gs_bucket}/{annotation_file.filename}"

    return ml_use, gs_path, tags[0].annotation_class.name


def _load_random_split(output_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    split_path = output_dir / "releases" / "latest" / "lists" / "split_v0_t0"

    with open(split_path / "random_train.txt", "r") as f:
        train_stems = [row.rstrip("\n") for row in f]

    with open(split_path / "random_val.txt", "r") as f:
        val_stems = [row.rstrip("\n") for row in f]

    with open(split_path / "random_test.txt", "r") as f:
        test_stems = [row.rstrip("\n") for row in f]

    return train_stems, val_stems, test_stems


def _infer_ml_use(filename: str, train_stems: List[str], val_stems: List[str], test_stems: List[str]) -> str:
    stem = Path(filename).stem

    if stem in train_stems:
        return "training"

    if stem in val_stems:
        return "validation"

    if stem in test_stems:
        return "test"

    raise ValueError(f"File with name {filename} does not belong to any partitions")
