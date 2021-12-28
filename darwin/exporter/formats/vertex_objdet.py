import csv
from pathlib import Path
from typing import Iterator, List, Tuple, cast

import darwin.datatypes as dt


def export(annotation_files: Iterator[dt.AnnotationFile], output_dir: Path) -> None:
    with open(output_dir.parent / "vertex.csv", "w") as f:
        for annotation_file in annotation_files:
            rows = export_file(annotation_file)
            for row in rows:
                writer = csv.writer(f)
                writer.writerow(row)


def export_file(
    annotation_file: dt.AnnotationFile,
) -> List[List[str]]:
    if annotation_file.is_video:
        return []

    bounding_boxes: List[dt.Annotation] = [
        cast(dt.Annotation, a)
        for a in annotation_file.annotations
        if a.annotation_class.annotation_type == "bounding_box"
    ]

    gs_bucket = "cloud-ai-platform-f304807d-674e-4e6e-aef2-d14213c2e0ff"
    gs_dataset_name = "bird-species"
    gs_path = f"gs://{gs_bucket}/{gs_dataset_name}/{annotation_file.filename}"

    if not annotation_file.image_width and not annotation_file.image_height:
        raise ValueError("Missing width and height information in annotation file")

    rows = []
    for bounding_box in bounding_boxes:
        xmin, ymin, xmax, ymax = _normalize_bounding_box(
            bounding_box=bounding_box,
            image_width=cast(int, annotation_file.image_width),
            image_height=cast(int, annotation_file.image_height),
        )
        rows.append(
            [
                gs_path,
                bounding_box.annotation_class.name,
                f"{xmin:.2f}",
                f"{ymin:.2f}",
                "",
                "",
                f"{xmax:.2f}",
                f"{ymax:.2f}",
                "",
                "",
            ]
        )
    return rows


def _normalize_bounding_box(
    bounding_box: dt.Annotation, image_width: int, image_height: int
) -> Tuple[float, float, float, float]:
    x = bounding_box.data["x"]
    y = bounding_box.data["y"]
    w = bounding_box.data["w"]
    h = bounding_box.data["h"]

    xmin = max(0, x) / image_width
    ymin = max(0, y) / image_height
    xmax = min(x + w, image_width) / image_width
    ymax = min(y + h, image_height) / image_height

    return xmin, ymin, xmax, ymax
