import json
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from darwin.dataset.download_manager import download_all_images_from_annotations
from darwin.dataset.identifier import DatasetIdentifier
from darwin.dataset.release import Release
from darwin.dataset.split_manager import split_dataset
from darwin.dataset.upload_manager import (
    FileUploadCallback,
    LocalFile,
    ProgressCallback,
    UploadHandler,
)
from darwin.dataset.utils import (
    exhaust_generator,
    get_annotations,
    get_classes,
    is_relative_to,
    is_unix_like_os,
    make_class_lists,
    sanitize_filename,
)
from darwin.datatypes import AnnotationClass, AnnotationFile, PathLike, Team
from darwin.exceptions import NotFound, UnsupportedExportFormat
from darwin.exporter.formats.darwin import build_image_annotation
from darwin.item import DatasetItem
from darwin.item_sorter import ItemSorter
from darwin.utils import find_files, parse_darwin_json, split_video_annotation, urljoin
from requests.models import Response
from rich.console import Console

if TYPE_CHECKING:
    from darwin.client import Client


class RemoteDataset:
    """
    Manages the remote and local versions of a dataset hosted on Darwin.
    It allows several dataset management operations such as syncing between
    remote and local, pulling a remote dataset, removing the local files, ...

    Parameters
    ----------
    client : Client
        Client to use for interaction with the server.
    team : str
        Team the dataset belongs to.
    name : str
        Name of the datasets as originally displayed on Darwin.
        It may contain white spaces, capital letters and special characters, e.g. `Bird Species!`.
    slug : str
        This is the dataset name with everything lower-case, removed specials characters and
        spaces are replaced by dashes, e.g., `bird-species`. This string is unique within a team.
    dataset_id : int
        Unique internal reference from the Darwin backend.
    item_count : int, default: 0
        Dataset size (number of items).
    progress : float, default: 0
        How much of the dataset has been annotated 0.0 to 1.0 (1.0 == 100%).

    Attributes
    ----------
    client : Client
        Client to use for interaction with the server.
    team : str
        Team the dataset belongs to.
    name : str
        Name of the datasets as originally displayed on Darwin.
        It may contain white spaces, capital letters and special characters, e.g. `Bird Species!`.
    slug : str
        This is the dataset name with everything lower-case, removed specials characters and
        spaces are replaced by dashes, e.g., `bird-species`. This string is unique within a team.
    dataset_id : int
        Unique internal reference from the Darwin backend.
    item_count : int, default: 0
        Dataset size (number of items).
    progress : float, default: 0
        How much of the dataset has been annotated 0.0 to 1.0 (1.0 == 100%).
    """

    def __init__(
        self,
        *,
        client: "Client",
        team: str,
        name: str,
        slug: str,
        dataset_id: int,
        item_count: int = 0,
        progress: float = 0,
    ):
        self.team = team
        self.name = name
        self.slug = slug or name
        self.dataset_id = dataset_id
        self.item_count = item_count
        self.progress = progress
        self.client = client
        self.annotation_types: Optional[List[Dict[str, Any]]] = None
        self.console: Console = Console()

    def push(
        self,
        files_to_upload: Optional[List[Union[PathLike, LocalFile]]],
        *,
        blocking: bool = True,
        multi_threaded: bool = True,
        fps: int = 0,
        as_frames: bool = False,
        files_to_exclude: Optional[List[PathLike]] = None,
        path: Optional[str] = None,
        preserve_folders: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        file_upload_callback: Optional[FileUploadCallback] = None,
    ) -> UploadHandler:
        """
        Uploads a local dataset (images ONLY) in the datasets directory.

        Parameters
        ----------
        files_to_upload : Optional[List[Union[PathLike, LocalFile]]]
            List of files to upload. Those can be folders.
        blocking : bool, default: True
            If False, the dataset is not uploaded and a generator function is returned instead.
        multi_threaded : bool, default: True
            Uses multiprocessing to upload the dataset in parallel.
            If blocking is False this has no effect.
        fps : int, default: 0
            When the uploading file is a video, specify its framerate.
        as_frames: bool, default: False
            When the uploading file is a video, specify whether it's going to be uploaded as a list of frames.
        files_to_exclude : Optional[PathLike]], default: None
            Optional list of files to exclude from the file scan. Those can be folders.
        path: Optional[str], default: None
            Optional path to store the files in.
        preserve_folders : bool, default: False
            Specify whether or not to preserve folder paths when uploading
        progress_callback: Optional[ProgressCallback], default: None
            Optional callback, called every time the progress of an uploading files is reported.
        file_upload_callback: Optional[FileUploadCallback], default: None
            Optional callback, called every time a file chunk is uploaded.

        Returns
        -------
        handler : UploadHandler
           Class for handling uploads, progress and error messages.

        Raises
        ------
        ValueError
            - If ``files_to_upload`` is ``None``.
            - If a path is specified when uploading a LocalFile object.
            - If there are no files to upload (because path is wrong or the exclude filter excludes everything).
        """

        if files_to_exclude is None:
            files_to_exclude = []

        if files_to_upload is None:
            raise ValueError("No files or directory specified.")

        uploading_files = [item for item in files_to_upload if isinstance(item, LocalFile)]
        search_files = [item for item in files_to_upload if not isinstance(item, LocalFile)]

        generic_parameters_specified = path is not None or fps != 0 or as_frames is not False
        if uploading_files and generic_parameters_specified:
            raise ValueError("Cannot specify a path when uploading a LocalFile object.")

        for found_file in find_files(search_files, files_to_exclude=files_to_exclude):
            local_path = path
            if preserve_folders:
                source_files = [source_file for source_file in search_files if is_relative_to(found_file, source_file)]
                if source_files:
                    local_path = str(found_file.relative_to(source_files[0]).parent)
            uploading_files.append(LocalFile(found_file, fps=fps, as_frames=as_frames, path=local_path))

        if not uploading_files:
            raise ValueError("No files to upload, check your path, exclusion filters and resume flag")

        handler = UploadHandler(self, uploading_files)
        if blocking:
            handler.upload(
                multi_threaded=multi_threaded,
                progress_callback=progress_callback,
                file_upload_callback=file_upload_callback,
            )
        else:
            handler.prepare_upload()

        return handler

    def split_video_annotations(self, release_name: str = "latest") -> None:
        """
        Splits the video annotations from this ``RemoteDataset`` using the given release.

        Parameters
        ----------
        release_name : str, default: "latest"
            The name of the release to use.
        """
        release_dir: Path = self.local_path / "releases" / release_name
        annotations_path: Path = release_dir / "annotations"

        for count, annotation_file in enumerate(annotations_path.glob("*.json")):
            darwin_annotation: Optional[AnnotationFile] = parse_darwin_json(annotation_file, count)
            if not darwin_annotation or not darwin_annotation.is_video:
                continue

            frame_annotations = split_video_annotation(darwin_annotation)
            for frame_annotation in frame_annotations:
                annotation = build_image_annotation(frame_annotation)

                video_frame_annotations_path = annotations_path / annotation_file.stem
                video_frame_annotations_path.mkdir(exist_ok=True, parents=True)

                stem = Path(frame_annotation.filename).stem
                output_path = video_frame_annotations_path / f"{stem}.json"
                with output_path.open("w") as f:
                    json.dump(annotation, f)

            # Finally delete video annotations
            annotation_file.unlink()

        # Update class list, which is used when loading local annotations in a dataset
        make_class_lists(release_dir)

    def pull(
        self,
        *,
        release: Optional[Release] = None,
        blocking: bool = True,
        multi_threaded: bool = True,
        only_annotations: bool = False,
        force_replace: bool = False,
        remove_extra: bool = False,
        subset_filter_annotations_function: Optional[Callable] = None,
        subset_folder_name: Optional[str] = None,
        use_folders: bool = False,
        video_frames: bool = False,
    ) -> Tuple[Optional[Callable[[], Iterator[Any]]], int]:
        """
        Downloads a remote dataset (images and annotations) to the datasets directory.

        Parameters
        ----------
        release: Optional[Release], default: None
            The release to pull.
        blocking : bool, default: True
            If False, the dataset is not downloaded and a generator function is returned instead.
        multi_threaded : bool, default: True
            Uses multiprocessing to download the dataset in parallel. If blocking is False this has no effect.
        only_annotations : bool, default: False
            Download only the annotations and no corresponding images.
        force_replace : bool, default: False
            Forces the re-download of an existing image.
        remove_extra : bool, default: False
            Removes existing images for which there is not corresponding annotation.
        subset_filter_annotations_function: Optional[Callable], default: None
            This function receives the directory where the annotations are downloaded and can
            perform any operation on them i.e. filtering them with custom rules or else.
            If it needs to receive other parameters is advised to use functools.partial() for it.
        subset_folder_name: Optional[str], default: None
            Name of the folder with the subset of the dataset. If not provided a timestamp is used.
        use_folders : bool, default: False
            Recreates folders from the dataset.
        video_frames : bool, default: False
            Pulls video frames images instead of video files.

        Returns
        -------
        generator : function
            Generator for doing the actual downloads. This is None if blocking is ``True``.
        count : int
            The number of files.

        Raises
        ------
        UnsupportedExportFormat
            If the given ``release`` has an invalid format.
        ValueError
            If darwin in unable to get ``Team`` configuration.
        """
        if release is None:
            release = self.get_release()

        if release.format != "json":
            raise UnsupportedExportFormat(release.format)

        release_dir = self.local_releases_path / release.name
        release_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            # Download the release from Darwin
            zip_file_path = release.download_zip(tmp_dir / "dataset.zip")
            with zipfile.ZipFile(zip_file_path) as z:
                # Extract annotations
                z.extractall(tmp_dir)
                # If a filtering function is provided, apply it
                if subset_filter_annotations_function is not None:
                    subset_filter_annotations_function(tmp_dir)
                    if subset_folder_name is None:
                        subset_folder_name = datetime.now().strftime("%m/%d/%Y_%H:%M:%S")
                annotations_dir: Path = release_dir / (subset_folder_name or "") / "annotations"
                # Remove existing annotations if necessary
                if annotations_dir.exists():
                    try:
                        shutil.rmtree(annotations_dir)
                    except PermissionError:
                        print(f"Could not remove dataset in {annotations_dir}. Permission denied.")
                annotations_dir.mkdir(parents=True, exist_ok=False)
                # Move the annotations into the right folder and rename them to have the image
                # original filename as contained in the json
                for annotation_path in tmp_dir.glob("*.json"):
                    with annotation_path.open() as file:
                        annotation = json.load(file)
                    filename = sanitize_filename(Path(annotation["image"]["filename"]).stem)
                    destination_name = annotations_dir / f"{filename}{annotation_path.suffix}"
                    shutil.move(str(annotation_path), str(destination_name))

        # Extract the list of classes and create the text files
        make_class_lists(release_dir)

        if release.latest and is_unix_like_os():
            try:
                latest_dir: Path = self.local_releases_path / "latest"
                if latest_dir.is_symlink():
                    latest_dir.unlink()

                target_link: Path = release_dir.name
                latest_dir.symlink_to(target_link)
            except OSError:
                self.console.log(f"Could not mark release {release.name} as latest. Continuing...")

        if only_annotations:
            # No images will be downloaded
            return None, 0

        team_config: Optional[Team] = self.client.config.get_team(self.team)
        if not team_config:
            raise ValueError("Unable to get Team configuration.")

        api_key = team_config.api_key

        # Create the generator with the download instructions
        progress, count = download_all_images_from_annotations(
            api_key=api_key,
            api_url=self.client.url,
            annotations_path=annotations_dir,
            images_path=self.local_images_path,
            force_replace=force_replace,
            remove_extra=remove_extra,
            use_folders=use_folders,
            video_frames=video_frames,
        )
        if count == 0:
            return None, count

        # If blocking is selected, download the dataset on the file system
        if blocking:
            exhaust_generator(progress=progress(), count=count, multi_threaded=multi_threaded)
            return None, count
        else:
            return progress, count

    def remove_remote(self) -> None:
        """Archives (soft-deletion) this ``RemoteDataset``."""
        self.client.archive_remote_dataset(self.dataset_id, self.team)

    def fetch_remote_files(
        self, filters: Optional[Dict[str, Union[str, List[str]]]] = None, sort: Optional[Union[str, ItemSorter]] = None
    ) -> Iterator[DatasetItem]:
        """
        Fetch and lists all files on the remote dataset.

        Parameters
        ----------
        filters : Optional[Dict[str, Union[str, List[str]]]], default: None
            The filters to use. Files excluded by the filter won't be fetched.
        sort : Optional[Union[str, ItemSorter]], default: None
            A sorting direction. It can be a string with the values 'asc', 'ascending', 'desc',
            'descending' or an ``ItemSorter`` instance.

        Yields
        -------
        Iterator[DatasetItem]
            An iterator of ``DatasetItem``.
        """
        post_filters: Dict[str, Union[str, List[str]]] = {}
        post_sort: Dict[str, str] = {}

        if filters:
            for list_type in ["filenames", "statuses"]:
                if list_type in filters:
                    if type(filters[list_type]) is list:
                        post_filters[list_type] = filters[list_type]
                    else:
                        post_filters[list_type] = str(filters[list_type])
            if "path" in filters:
                post_filters["path"] = str(filters["path"])
            if "types" in filters:
                post_filters["types"] = str(filters["types"])

            if sort:
                item_sorter = ItemSorter.parse(sort)
                post_sort[item_sorter.field] = item_sorter.direction.value
        cursor = {"page[size]": 500}
        while True:
            payload = {"filter": post_filters, "sort": post_sort}
            response = self.client.fetch_remote_files(self.dataset_id, cursor, payload, self.team)

            yield from [DatasetItem.parse(item) for item in response["items"]]

            if response["metadata"]["next"]:
                cursor["page[from]"] = response["metadata"]["next"]
            else:
                return

    def archive(self, items: Iterator[DatasetItem]) -> None:
        """
        Archives (soft-deletion) the given ``DatasetItem``s belonging to this ``RemoteDataset``.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``s to be archived.
        """
        payload: Dict[str, Any] = {"filter": {"dataset_item_ids": [item.id for item in items]}}
        self.client.archive_item(self.slug, self.team, payload)

    def restore_archived(self, items: Iterator[DatasetItem]) -> None:
        """
        Restores the archived ``DatasetItem``s that belong to this ``RemoteDataset``.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``s to be restored.
        """
        payload: Dict[str, Any] = {"filter": {"dataset_item_ids": [item.id for item in items]}}
        self.client.restore_archived_item(self.slug, self.team, payload)

    def move_to_new(self, items: Iterator[DatasetItem]) -> None:
        """
        Changes the given ``DatasetItem``s status to ``new``.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``s whose status will change.
        """
        payload: Dict[str, Any] = {"filter": {"dataset_item_ids": [item.id for item in items]}}
        self.client.move_item_to_new(self.slug, self.team, payload)

    def reset(self, items: Iterator[DatasetItem]) -> None:
        """
        Resets the given ``DatasetItem``s.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``s to be resetted.
        """
        payload: Dict[str, Any] = {"filter": {"dataset_item_ids": [item.id for item in items]}}
        self.client.reset_item(self.slug, self.team, payload)

    def delete_items(self, items: Iterator[DatasetItem]) -> None:
        """
        Deletes the given ``DatasetItem``s.

        Parameters
        ----------
        items : Iterator[DatasetItem]
            The ``DatasetItem``s to be deleted.
        """
        payload: Dict[str, Any] = {"filter": {"dataset_item_ids": [item.id for item in items]}}
        self.client.delete_item(self.slug, self.team, payload)

    def fetch_annotation_type_id_for_name(self, name: str) -> Optional[int]:
        """
        Fetches annotation type id for a annotation type name, such as ``bounding_box``.

        Parameters
        ----------
        name: str
            The name of the annotation we want the id for.


        Returns
        -------
        Optional[int]
            The id of the annotation type or ``None`` if it doesn't exist.
        """
        if not self.annotation_types:
            self.annotation_types = self.client.annotation_types()

        for annotation_type in self.annotation_types:
            if annotation_type["name"] == name:
                return annotation_type["id"]

        return None

    def create_annotation_class(self, name: str, type: str, subtypes: List[str] = []) -> Dict[str, Any]:
        """
        Creates an annotation class for this ``RemoteDataset``.

        Parameters
        ----------
        name : str
            The name of the annotation class.
        type : str
            The type of the annotation class.
        subtypes : List[str], default: []
            Annotation class subtypes.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the server response.

        Raises
        ------
        ValueError
            If a given annotation type or subtype is unknown.
        """

        type_ids: List[int] = []
        for annotation_type in [type] + subtypes:
            type_id: Optional[int] = self.fetch_annotation_type_id_for_name(annotation_type)
            if not type_id and self.annotation_types is not None:
                list_of_annotation_types = ", ".join([type["name"] for type in self.annotation_types])
                raise ValueError(
                    f"Unknown annotation type: '{annotation_type}', valid values: {list_of_annotation_types}"
                )

            if type_id is not None:
                type_ids.append(type_id)

        return self.client.create_annotation_class(self.dataset_id, type_ids, name)

    def add_annotation_class(self, annotation_class: Union[AnnotationClass, int]) -> Optional[Dict[str, Any]]:
        """
        Adds an annotation class to this ``RemoteDataset``.

        Parameters
        ----------
        annotation_class : Union[AnnotationClass, int]
            The annotation class to add or its id.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary with the server response or ``None`` if the annotations class already
            exists.

        Raises
        ------
        ValueError
            If the given ``annotation_class`` does not exist in this ``RemoteDataset``'s team.
        """
        # Waiting for a better api for setting classes
        # in the meantime this will do
        all_classes: List[Dict[str, Any]] = self.fetch_remote_classes(True)

        if isinstance(annotation_class, int):
            match = [cls for cls in all_classes if cls["id"] == annotation_class]
            if not match:
                raise ValueError(f"Annotation class id: `{annotation_class}` does not exist in Team.")
        else:
            annotation_class_type = annotation_class.annotation_internal_type or annotation_class.annotation_type
            match = [
                cls
                for cls in all_classes
                if cls["name"] == annotation_class.name and annotation_class_type in cls["annotation_types"]
            ]
            if not match:
                # We do not expect to reach here; as pervious logic divides annotation classes in imports
                # between `in team` and `new to platform`
                raise ValueError(
                    f"Annotation class name: `{annotation_class.name}`, type: `{annotation_class_type}`; does not exist in Team."
                )

        datasets = match[0]["datasets"]
        # check that we are not already part of the dataset
        for dataset in datasets:
            if dataset["id"] == self.dataset_id:
                return None
        datasets.append({"id": self.dataset_id})
        # we typecast to dictionary because we are not passing the raw=True parameter.
        class_id = match[0]["id"]
        payload = {"datasets": datasets, "id": class_id}
        return self.client.update_annotation_class(class_id, payload)

    def fetch_remote_classes(self, team_wide=False) -> List[Dict[str, Any]]:
        """
        Fetches all the Annotation Classes from this ``RemoteDataset``.

        Parameters
        ----------
        team_wide : bool, default: False
            If ``True`` will return all Annotation Classes that belong to the team. If ``False``
            will only return Annotation Classes which have been added to the dataset.

        Returns
        -------
        List[Dict[str, Any]]:
            List of Annotation Classes (can be empty).
        """
        all_classes: List[Dict[str, Any]] = self.client.fetch_remote_classes()

        classes_to_return = []
        for cls in all_classes:
            belongs_to_current_dataset = any([dataset["id"] == self.dataset_id for dataset in cls["datasets"]])
            cls["available"] = belongs_to_current_dataset
            if team_wide or belongs_to_current_dataset:
                classes_to_return.append(cls)
        return classes_to_return

    def fetch_remote_attributes(self) -> List[Dict[str, Any]]:
        """
        Fetches all remote attributes on the remote dataset.

        Returns
        -------
        List[Dict[str, Any]]
            A List with the attributes, where each attribute is a dictionary.
        """
        return self.client.fetch_remote_attributes(self.dataset_id)

    def export(
        self,
        name: str,
        annotation_class_ids: Optional[List[str]] = None,
        include_url_token: bool = False,
        include_authorship: bool = False,
    ) -> None:
        """
        Create a new release for this ``RemoteDataset``.

        Parameters
        ----------
        name : str
            Name of the release.
        annotation_class_ids : Optional[List[str]], default: None
            List of the classes to filter.
        include_url_token : bool, default: False
            Should the image url in the export include a token enabling access without team
            membership or not?
        include_authorship : bool, default: False
            If set, include annotator and reviewer metadata for each annotation.

        """
        if annotation_class_ids is None:
            annotation_class_ids = []

        payload = {
            "annotation_class_ids": annotation_class_ids,
            "name": name,
            "include_export_token": include_url_token,
            "include_authorship": include_authorship,
        }
        self.client.create_export(self.dataset_id, payload, self.team)

    def get_report(self, granularity: str = "day") -> str:
        """
        Returns a String representation of a CSV report for this ``RemoteDataset``.

        Parameters
        ----------
        granularity : str, default: "day"
            The granularity of the report, can be 'day', 'week' or 'month'.

        Returns
        -------
        str
            A CSV report.
        """
        response: Response = self.client.get_report(self.dataset_id, granularity, self.team)
        return response.text

    def get_releases(self) -> List["Release"]:
        """
        Get a sorted list of releases with the most recent first.

        Returns
        -------
        List["Release"]
            Returns a sorted list of available ``Release``s with the most recent first.
        """
        try:
            releases_json: List[Dict[str, Any]] = self.client.get_exports(self.dataset_id, self.team)
        except NotFound:
            return []

        releases = [Release.parse_json(self.slug, self.team, payload) for payload in releases_json]
        return sorted(filter(lambda x: x.available, releases), key=lambda x: x.version, reverse=True)

    def get_release(self, name: str = "latest") -> "Release":
        """
        Get a specific ``Release`` for this ``RemoteDataset``.

        Parameters
        ----------
        name : str, default: "latest"
            Name of the export.

        Returns
        -------
        Release
            The selected release.

        Raises
        ------
        NotFound
            The selected ``Release`` does not exist.
        """
        releases = self.get_releases()
        if not releases:
            raise NotFound(str(self.identifier))

        if name == "latest":
            return next((release for release in releases if release.latest))

        for release in releases:
            if str(release.name) == name:
                return release
        raise NotFound(str(self.identifier))

    def split(
        self,
        val_percentage: float = 0.1,
        test_percentage: float = 0,
        split_seed: int = 0,
        make_default_split: bool = True,
        release_name: Optional[str] = None,
    ) -> None:
        """
        Creates lists of file names for each split for train, validation, and test.
        Note: This functions needs a local copy of the dataset.

        Parameters
        ----------
        val_percentage : float, default: 0.1
            Percentage of images used in the validation set.
        test_percentage : float, default: 0
            Percentage of images used in the test set.
        split_seed : int, default: 0
            Fix seed for random split creation.
        make_default_split: bool, default: True
            Makes this split the default split.
        release_name: Optional[str], default: None
            Version of the dataset.

        Raises
        ------
        NotFound
            If this ``RemoteDataset`` is not found locally.
        """
        if not self.local_path.exists():
            raise NotFound(
                "Local dataset not found: the split is performed on the local copy of the dataset. \
                           Pull the dataset from Darwin first using pull()"
            )
        if release_name in ["latest", None]:
            release = self.get_release("latest")
            release_name = release.name

        split_dataset(
            self.local_path,
            release_name=release_name,
            val_percentage=val_percentage,
            test_percentage=test_percentage,
            split_seed=split_seed,
            make_default_split=make_default_split,
        )

    def classes(self, annotation_type: str, release_name: Optional[str] = None) -> List[str]:
        """
        Returns the list of ``class_type`` classes.

        Parameters
        ----------
        annotation_type : str
            The type of annotation classes, e.g. 'tag' or 'polygon'.
        release_name: Optional[str], default: None
            Version of the dataset.


        Returns
        -------
        classes: List[str]
            List of classes in the dataset of type ``class_type``.
        """
        assert self.local_path.exists()
        if release_name in ["latest", None]:
            release = self.get_release("latest")
            release_name = release.name

        return get_classes(self.local_path, release_name=release_name, annotation_type=annotation_type)

    def annotations(
        self,
        partition: str,
        split: str = "split",
        split_type: str = "stratified",
        annotation_type: str = "polygon",
        release_name: Optional[str] = None,
        annotation_format: Optional[str] = "darwin",
    ) -> Iterable[Dict[str, Any]]:
        """
        Returns all the annotations of a given split and partition in a single dictionary.

        Parameters
        ----------
        partition : str
            Selects one of the partitions [train, val, test].
        split : str, default: "split"
            Selects the split that defines the percentages used (use 'split' to select the default split.
        split_type : str, default: "stratified"
            Heuristic used to do the split [random, stratified].
        annotation_type : str, default: "polygon"
            The type of annotation classes [tag, polygon].
        release_name : Optional[str], default: None
            Version of the dataset.
        annotation_format : Optional[str], default: "darwin"
            Re-formatting of the annotation when loaded [coco, darwin].

        Yields
        -------
        Dict[str, Any]
            Dictionary representing an annotation from this ``RemoteDataset``.
        """
        assert self.local_path.exists()
        if release_name in ["latest", None]:
            release = self.get_release("latest")
            release_name = release.name

        for annotation in get_annotations(
            self.local_path,
            partition=partition,
            split=split,
            split_type=split_type,
            annotation_type=annotation_type,
            release_name=release_name,
            annotation_format=annotation_format,
        ):
            yield annotation

    def workview_url_for_item(self, item: DatasetItem) -> str:
        """
        Returns the darwin URL for the given ``DatasetItem``.

        Parameters
        ----------
        item : DatasetItem
            The ``DatasetItem`` for which we want the url.

        Returns
        -------
        str
            The url.
        """
        return urljoin(self.client.base_url, f"/workview?dataset={self.dataset_id}&image={item.seq}")

    @property
    def remote_path(self) -> Path:
        """Returns an URL specifying the location of the remote dataset."""
        return Path(urljoin(self.client.base_url, f"/datasets/{self.dataset_id}"))

    @property
    def local_path(self) -> Path:
        """Returns a Path to the local dataset."""
        datasets_dir: str = self.client.get_datasets_dir(self.team)

        if self.slug:
            return Path(datasets_dir) / self.team / self.slug
        else:
            return Path(datasets_dir) / self.team

    @property
    def local_releases_path(self) -> Path:
        """Returns a Path to the local dataset releases."""
        return self.local_path / "releases"

    @property
    def local_images_path(self) -> Path:
        """Returns a local Path to the images folder."""
        return self.local_path / "images"

    @property
    def identifier(self) -> DatasetIdentifier:
        """The ``DatasetIdentifier`` of this ``RemoteDataset``."""
        return DatasetIdentifier(team_slug=self.team, dataset_slug=self.slug)
