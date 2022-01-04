import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import boto3
import darwin.exporter as exporter
import google.cloud.aiplatform as aiplatform
import google.cloud.storage as storage
import proto
from darwin.dataset.download_manager import download_all_images_from_annotations
from darwin.dataset.utils import exhaust_generator
from darwin.datatypes import ExportParser
from darwin.exporter.formats.vertex_objdet import export as object_detection_parser
from darwin.exporter.formats.vertex_objtrack import export as object_tracking_parser
from darwin.exporter.formats.vertex_slc import export as single_label_parser
from google.api_core.client_options import ClientOptions
from google.cloud.aiplatform.gapic.schema import trainingjob
from google.cloud.aiplatform_v1.types.training_pipeline import TrainingPipeline
from google.cloud.storage.bucket import Bucket
from rich.console import Console
from rich.progress import track

console = Console()


@dataclass(frozen=True)
class VertexConfig:
    api_key: str
    api_url: str
    darwin_bucket: str
    export_key: str
    export_name: str
    export_parser: ExportParser
    import_schema_uri: str
    milli_node_hours: int
    model_type: str
    training_job_definition: proto.Message
    training_task_definition: str


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


def download_images(api_key: str, api_url: str, annotations_path: Path, images_path: Path) -> None:
    progress, count = download_all_images_from_annotations(
        api_key=api_key,
        api_url=api_url,
        annotations_path=annotations_path,
        images_path=images_path,
    )

    console.print(f"Downloading {count} images...")
    exhaust_generator(progress=progress(), count=count, multi_threaded=True)


def push_images(images_path: Path, export_name: str, bucket: Bucket) -> None:
    image_paths = list(images_path.glob("*"))
    for image_path in track(image_paths):
        blob = bucket.blob(f"{export_name}/{image_path.name}")
        import_filename = str(image_path)
        blob.upload_from_filename(import_filename)


def push_annotations(remote_name: str, dataset_path: Path, bucket: Bucket) -> None:
    blob = bucket.blob(remote_name)
    import_filename = "vertex.csv"
    blob.upload_from_filename(str(dataset_path.parent / import_filename))
    print(f"File {remote_name} uploaded to {bucket.name}")


def create_vertex_dataset(config: VertexConfig, bucket: Bucket, remote_name: str, location: str) -> None:
    ds = aiplatform.ImageDataset.create(
        display_name=config.export_name,
        gcs_source=f"gs://{bucket.name}/{remote_name}",
        import_schema_uri=config.import_schema_uri,
        location=location,
        sync=True,
    )
    ds.wait()
    console.print(f"Vertex AI Dataset {ds.display_name} created successfully as {ds.resource_name}")


def create_training_pipeline(
    dataset_id: str, project_name: str, location: str, config: VertexConfig
) -> TrainingPipeline:
    client_options = ClientOptions(api_endpoint=f"{location}-aiplatform.googleapis.com")
    client = aiplatform.gapic.PipelineServiceClient(client_options=client_options)

    training_task_inputs = config.training_job_definition(
        model_type=config.model_type,
        budget_milli_node_hours=config.milli_node_hours,
        disable_early_stopping=False,
    ).to_value()

    training_pipeline = TrainingPipeline(
        display_name=f"{config.export_name}-pipeline",
        training_task_definition=config.training_task_definition,
        training_task_inputs=training_task_inputs,
        input_data_config={"dataset_id": dataset_id},
        model_to_upload={"display_name": f"{config.export_name}-model"},
    )

    parent = f"projects/{project_name}/locations/{location}"
    return client.create_training_pipeline(parent=parent, training_pipeline=training_pipeline)


configs = {
    "single-label-classification": VertexConfig(
        api_key="4xJxe-S.2TNKezgz1rycV1j_7a0pUL8JVhGZDxHc",
        api_url="https://staging.v7labs.com/api/",
        darwin_bucket="njord-experiments",
        export_key="exports/andreas-team/bird-species/andreas-team/bird-species@auto-1640014413.zip",
        export_name="bird-species-tags",
        export_parser=single_label_parser,
        import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification,
        milli_node_hours=8000,
        model_type="CLOUD",
        training_job_definition=trainingjob.definition.AutoMlImageClassificationInputs,
        training_task_definition="gs://google-cloud-aiplatform/schema/trainingjob/definition/automl_image_classification_1.0.0.yaml",
    ),
    "object-detection": VertexConfig(
        api_key="4xJxe-S.2TNKezgz1rycV1j_7a0pUL8JVhGZDxHc",
        api_url="https://staging.v7labs.com/api/",
        darwin_bucket="njord-experiments",
        export_key="exports/andreas-team/bird-species-boxes/andreas-team/bird-species-boxes@auto-1632929310.zip",
        export_name="bird-species-bounding-boxes",
        export_parser=object_detection_parser,
        import_schema_uri=aiplatform.schema.dataset.ioformat.image.bounding_box,
        milli_node_hours=20000,
        model_type="CLOUD_HIGH_ACCURACY_1",
        training_job_definition=trainingjob.definition.AutoMlImageObjectDetectionInputs,
        training_task_definition="gs://google-cloud-aiplatform/schema/trainingjob/definition/automl_image_object_detection_1.0.0.yaml",
    ),
    "object-tracking": VertexConfig(
        api_key="93lt1gU.JWL6-OaiYEWgBq1QrYclbGxvq3-NeVRA",
        api_url="https://darwin.v7labs.com/api/",
        darwin_bucket="darwin-production-data",
        export_key="exports/rockfishvision/accesscontrol-test-videos/test-v0.0.3.zip",
        export_name="rv-people",
        export_parser=object_tracking_parser,
    ),
}


if __name__ == "__main__":
    config = configs["object-tracking"]

    # 1. Download Darwin dataset export from AWS

    dataset_path = download_export(
        bucket_name=config.darwin_bucket, export_key=config.export_key, name=config.export_name
    )

    images_path = dataset_path / "images"
    annotations_path = dataset_path / "releases" / "latest" / "annotations"

    # 2. Download all images beloging to that export

    download_images(
        api_key=config.api_key, api_url=config.api_url, annotations_path=annotations_path, images_path=images_path
    )

    # 3. Push images to Google Cloud Storage

    storage_client = storage.Client()
    bucket = storage_client.bucket("cloud-ai-platform-f304807d-674e-4e6e-aef2-d14213c2e0ff")
    push_images(images_path=images_path, export_name=config.export_name, bucket=bucket)

    # 4. Convert annotations to Vertex CSV

    exporter.export_annotations(config.export_parser, [annotations_path], dataset_path)

    # 5. Upload annotations to Google Cloud Storage

    remote_name = f"{config.export_name}.csv"
    location = "europe-west4"
    project_name = "prefab-icon-336200"

    push_annotations(remote_name=remote_name, dataset_path=dataset_path, bucket=bucket)

    # 6. Create Vertex Dataset

    create_vertex_dataset(config=config, bucket=bucket, remote_name=remote_name, location=location)

    # 7. Create Training Pipeline

    dataset_index = {d.display_name: d.name for d in aiplatform.ImageDataset.list(location=location)}
    dataset_id = dataset_index[config.export_name]

    response = create_training_pipeline(
        dataset_id=dataset_id, project_name=project_name, location=location, config=config
    )
    print("response:", response)

    # client.get_training_pipeline(name="projects/125473047028/locations/europe-west4/trainingPipelines/2107398752586170368")
