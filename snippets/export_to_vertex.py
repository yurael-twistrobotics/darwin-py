"""
Fetch a Darwin export, pull the data, split it into train/val/test partitions,
convert them into vertex format (single-label classification), push it to GCP.
"""

import tempfile
import zipfile
from pathlib import Path

import boto3
import darwin.exporter as exporter
from darwin.dataset.download_manager import download_all_images_from_annotations
from darwin.dataset.split_manager import split_dataset
from darwin.dataset.utils import exhaust_generator
from darwin.exporter.formats.vertex_slc import export as parser
from rich.console import Console

console = Console()


def download_export(bucket_name: str, export_key: str, name: str) -> Path:
    tmpdir = Path(tempfile.gettempdir())
    output_path = str(tmpdir / f"{name}.zip")

    try:
        s3 = boto3.resource("s3")
        s3.Bucket(bucket_name).download_file(export_key, output_path)
    finally:
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir / name / "releases" / "latest" / "annotations")

    console.print(f"Export downloaded and exported successfully. Find it at {tmpdir}")

    return tmpdir / name


if __name__ == "__main__":
    api_key = "4xJxe-S.2TNKezgz1rycV1j_7a0pUL8JVhGZDxHc"
    api_url = "https://staging.v7labs.com/api/"

    bucket_name = "njord-experiments"
    export_key = "exports/andreas-team/bird-species/andreas-team/bird-species@auto-1640014413.zip"
    export_name = "bird-species"

    dataset_path = download_export(bucket_name=bucket_name, export_key=export_key, name=export_name)
    images_path = dataset_path / "images"
    annotations_path = dataset_path / "releases" / "latest" / "annotations"

    progress, count = download_all_images_from_annotations(
        api_key=api_key,
        api_url=api_url,
        annotations_path=annotations_path,
        images_path=images_path,
    )

    console.print(f"Downloading {count} images...")
    exhaust_generator(progress=progress(), count=count, multi_threaded=True)

    split_path = split_dataset(dataset_path=dataset_path, val_percentage=0.1, test_percentage=0.1)
    console.print(f"Dataset {Path(export_key).name} has been split here: {split_path}")

    exporter.export_annotations(parser, [annotations_path], dataset_path)
