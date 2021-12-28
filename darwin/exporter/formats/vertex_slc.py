import csv
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import darwin.datatypes as dt


def export(annotation_files: Iterator[dt.AnnotationFile], output_dir: Path) -> None:
    with open(output_dir.parent / "vertex.csv", "w") as f:
        for annotation_file in annotation_files:
            row = export_file(annotation_file)
            if row is None:
                continue

            writer = csv.writer(f)
            writer.writerow(row)


def export_file(annotation_file: dt.AnnotationFile) -> Optional[Tuple[str, str]]:
    if annotation_file.is_video:
        return None

    tags = [a for a in annotation_file.annotations if a.annotation_class.annotation_type == "tag"]
    if len(tags) != 1:
        return None

    gs_bucket = "cloud-ai-platform-f304807d-674e-4e6e-aef2-d14213c2e0ff"
    gs_path = f"gs://{gs_bucket}/darwin/{annotation_file.filename}"

    return gs_path, tags[0].annotation_class.name
