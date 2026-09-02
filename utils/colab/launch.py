from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from threading import Lock, local
from typing import Any, Callable, TypeVar


FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
DEFAULT_TOKEN_PATH = Path(__file__).resolve().parent / "google_drive_token.json"
FIXED_EXCLUDE_PATTERNS = (
    ".git/",
    "__pycache__/",
    "*.pyc",
    ".gitignore",
    ".dockerignore",
    ".ignore",
    ".rgignore",
    "weights/",
    "/eval/results"
    ".env"
)
DEFAULT_SYNC_WORKERS = 8
PROGRESS_LABEL_WIDTH = 60

T = TypeVar("T")


@dataclass(frozen=True)
class LocalFile:
    rel_path: str
    abs_path: Path
    size: int


@dataclass(frozen=True)
class RemoteEntry:
    id: str
    rel_path: str
    name: str
    parent_id: str | None
    mime_type: str

    @property
    def is_folder(self) -> bool:
        return self.mime_type == FOLDER_MIME_TYPE


@dataclass(frozen=True)
class LocalScanResult:
    files: dict[str, LocalFile]
    ignored_paths: tuple[str, ...]


@dataclass(frozen=True)
class UpdateTask:
    local_file: LocalFile
    remote_entry: RemoteEntry


@dataclass(frozen=True)
class SyncPlan:
    uploads: tuple[LocalFile, ...]
    updates: tuple[UpdateTask, ...]
    deletes: tuple[RemoteEntry, ...]
    duplicate_deletes: tuple[RemoteEntry, ...]
    preserved_remote: tuple[RemoteEntry, ...]


class IgnoreMatcher:
    def __init__(self, patterns: list[str]):
        self.patterns = [pattern for pattern in patterns if pattern]
        self._pathspec = self._build_pathspec(self.patterns)

    @staticmethod
    def _build_pathspec(patterns: list[str]) -> Any | None:
        try:
            import pathspec
        except ImportError:
            return None
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def matches(self, rel_path: str, *, is_dir: bool = False) -> bool:
        normalized = normalize_rel_path(rel_path)
        if not normalized:
            return False

        candidate = f"{normalized}/" if is_dir else normalized

        if self._pathspec is not None:
            return self._pathspec.match_file(candidate)

        return _matches_gitignore_patterns(candidate, self.patterns)


def normalize_rel_path(rel_path: str | Path) -> str:
    raw = str(rel_path).replace("\\", "/").strip("/")
    if not raw or raw == ".":
        return ""
    return str(PurePosixPath(raw))


def load_ignore_patterns(repo_root: Path) -> list[str]:
    patterns = list(FIXED_EXCLUDE_PATTERNS)
    gitignore_path = repo_root / ".gitignore"
    if not gitignore_path.exists():
        return patterns

    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    return patterns


def collect_local_files(
    repo_root: Path,
    ignore_matcher: IgnoreMatcher | None = None,
) -> LocalScanResult:
    matcher = ignore_matcher or IgnoreMatcher(load_ignore_patterns(repo_root))
    files: dict[str, LocalFile] = {}
    ignored: list[str] = []

    for current_root, dirnames, filenames in os.walk(repo_root):
        current_path = Path(current_root)
        rel_dir = normalize_rel_path(current_path.relative_to(repo_root))

        kept_dirs: list[str] = []
        for dirname in sorted(dirnames):
            rel_path = normalize_rel_path(PurePosixPath(rel_dir, dirname))
            if matcher.matches(rel_path, is_dir=True):
                ignored.append(f"{rel_path}/")
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in sorted(filenames):
            rel_path = normalize_rel_path(PurePosixPath(rel_dir, filename))
            if matcher.matches(rel_path):
                ignored.append(rel_path)
                continue

            abs_path = repo_root / rel_path
            files[rel_path] = LocalFile(
                rel_path=rel_path,
                abs_path=abs_path,
                size=abs_path.stat().st_size,
            )

    return LocalScanResult(files=files, ignored_paths=tuple(sorted(set(ignored))))


def parse_dest_path(dest_path: str) -> list[str]:
    segments = [segment.strip() for segment in dest_path.replace("\\", "/").split("/")]
    clean_segments = [segment for segment in segments if segment]
    if clean_segments and clean_segments[0] == "MyDrive":
        clean_segments = clean_segments[1:]

    if not clean_segments:
        raise ValueError("`--dest-path` must include at least one folder name.")
    if any(segment in {".", ".."} for segment in clean_segments):
        raise ValueError("`--dest-path` cannot include `.` or `..` segments.")
    return clean_segments


def build_sync_plan(
    local_files: dict[str, LocalFile],
    remote_files: dict[str, RemoteEntry],
    remote_folders: dict[str, RemoteEntry],
    remote_duplicates: tuple[RemoteEntry, ...],
    ignore_matcher: IgnoreMatcher,
) -> SyncPlan:
    ignored_folder_prefixes = {
        rel_path
        for rel_path, entry in remote_folders.items()
        if ignore_matcher.matches(rel_path, is_dir=True)
    }

    local_parent_paths = {
        parent_path
        for local_file in local_files.values()
        for parent_path in iter_parent_paths(local_file.rel_path)
    }
    preserved_paths: set[str] = set(local_parent_paths)
    # Folder deletions are recursive in Drive.  Compute ignored-folder
    # ancestors before evaluating folders below; their lexical traversal is
    # parent-first, so discovering a protected child later is too late.
    for rel_path, entry in remote_folders.items():
        if should_preserve_remote_path(
            rel_path,
            is_dir=entry.is_folder,
            ignore_matcher=ignore_matcher,
            ignored_folder_prefixes=ignored_folder_prefixes,
        ):
            preserved_paths.add(rel_path)
            preserved_paths.update(iter_parent_paths(rel_path))
    preserved_entries: list[RemoteEntry] = []
    upload_files: list[LocalFile] = []
    update_tasks: list[UpdateTask] = []
    deletable_files: list[RemoteEntry] = []
    duplicate_deletes: list[RemoteEntry] = []
    deletable_folders: list[RemoteEntry] = []

    for rel_path, local_file in sorted(local_files.items()):
        remote_entry = remote_files.get(rel_path)
        if remote_entry is None:
            upload_files.append(local_file)
            continue
        update_tasks.append(UpdateTask(local_file=local_file, remote_entry=remote_entry))

    for entry in sorted(remote_duplicates, key=lambda item: item.rel_path):
        if (
            entry.is_folder
            or entry.rel_path not in local_files
            or should_preserve_remote_path(
                entry.rel_path,
                is_dir=entry.is_folder,
                ignore_matcher=ignore_matcher,
                ignored_folder_prefixes=ignored_folder_prefixes,
            )
        ):
            preserved_paths.add(entry.rel_path)
            preserved_paths.update(iter_parent_paths(entry.rel_path))
            preserved_entries.append(entry)
            continue
        duplicate_deletes.append(entry)

    for rel_path, entry in sorted(remote_files.items()):
        if rel_path in local_files:
            continue
        if should_preserve_remote_path(
            rel_path,
            is_dir=False,
            ignore_matcher=ignore_matcher,
            ignored_folder_prefixes=ignored_folder_prefixes,
        ):
            preserved_paths.add(rel_path)
            preserved_paths.update(iter_parent_paths(rel_path))
            preserved_entries.append(entry)
            continue
        deletable_files.append(entry)

    for rel_path, entry in sorted(remote_folders.items()):
        if should_preserve_remote_path(
            rel_path,
            is_dir=True,
            ignore_matcher=ignore_matcher,
            ignored_folder_prefixes=ignored_folder_prefixes,
        ) or rel_path in preserved_paths:
            preserved_paths.add(rel_path)
            preserved_paths.update(iter_parent_paths(rel_path))
            preserved_entries.append(entry)
            continue
        deletable_folders.append(entry)

    deletable_folders.sort(key=lambda entry: entry.rel_path.count("/"), reverse=True)

    return SyncPlan(
        uploads=tuple(upload_files),
        updates=tuple(update_tasks),
        deletes=tuple(deletable_files + deletable_folders),
        duplicate_deletes=tuple(duplicate_deletes),
        preserved_remote=tuple(
            sorted(preserved_entries, key=lambda entry: (entry.rel_path, entry.is_folder))
        ),
    )


def should_preserve_remote_path(
    rel_path: str,
    *,
    is_dir: bool,
    ignore_matcher: IgnoreMatcher,
    ignored_folder_prefixes: set[str],
) -> bool:
    if ignore_matcher.matches(rel_path, is_dir=is_dir):
        return True
    return any(
        rel_path == prefix or rel_path.startswith(f"{prefix}/")
        for prefix in ignored_folder_prefixes
    )


def iter_parent_paths(rel_path: str) -> tuple[str, ...]:
    parents: list[str] = []
    current = normalize_rel_path(PurePosixPath(rel_path).parent)
    while current:
        parents.append(current)
        current = normalize_rel_path(PurePosixPath(current).parent)
    return tuple(parents)


def validate_client_secret_file(client_secret_path: Path) -> None:
    if not client_secret_path.exists():
        raise FileNotFoundError(f"Client secret file not found: {client_secret_path}")
    if client_secret_path.stat().st_size == 0:
        raise ValueError(
            f"Client secret file is empty: {client_secret_path}. "
            "Download the OAuth Desktop app JSON from Google Cloud Console."
        )

    try:
        payload = json.loads(client_secret_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Client secret file is not valid JSON: {client_secret_path}"
        ) from exc

    if "installed" not in payload:
        top_keys = ", ".join(sorted(payload.keys()))
        raise ValueError(
            "Client secret JSON must be a Desktop app OAuth credential with an "
            f"`installed` section. Found keys: {top_keys or '(none)'}."
        )


def get_drive_service(client_secret_path: Path, token_path: Path) -> Any:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive dependencies are missing. "
            "Install them with `pip install -r requirements.launch.txt`."
        ) from exc

    validate_client_secret_file(client_secret_path)

    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(
            str(token_path),
            DRIVE_SCOPES,
        )

    if credentials is None or not credentials.valid:
        if credentials is not None and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_secret_path),
                DRIVE_SCOPES,
            )
            credentials = flow.run_local_server(port=0)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(credentials.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def ensure_drive_folder(service: Any, parent_id: str, folder_name: str) -> str:
    response = service.files().list(
        q=(
            f"trashed = false and mimeType = '{FOLDER_MIME_TYPE}' "
            f"and name = '{escape_drive_query_value(folder_name)}' "
            f"and '{parent_id}' in parents"
        ),
        spaces="drive",
        fields="files(id, name, mimeType, parents)",
        pageSize=10,
    ).execute()
    existing = response.get("files", [])
    if existing:
        return existing[0]["id"]

    created = service.files().create(
        body={
            "name": folder_name,
            "mimeType": FOLDER_MIME_TYPE,
            "parents": [parent_id],
        },
        fields="id",
    ).execute()
    return created["id"]


def find_drive_folder(service: Any, parent_id: str, folder_name: str) -> str | None:
    response = service.files().list(
        q=(
            f"trashed = false and mimeType = '{FOLDER_MIME_TYPE}' "
            f"and name = '{escape_drive_query_value(folder_name)}' "
            f"and '{parent_id}' in parents"
        ),
        spaces="drive",
        fields="files(id, name, mimeType, parents)",
        pageSize=10,
    ).execute()
    existing = response.get("files", [])
    if not existing:
        return None
    return existing[0]["id"]


def resolve_dest_root(
    service: Any,
    dest_segments: list[str],
    *,
    create: bool = True,
) -> str | None:
    parent_id = "root"
    for segment in dest_segments:
        if create:
            parent_id = ensure_drive_folder(service, parent_id, segment)
            continue

        existing_id = find_drive_folder(service, parent_id, segment)
        if existing_id is None:
            return None
        parent_id = existing_id
    return parent_id


def list_children(service: Any, parent_id: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page_token: str | None = None

    while True:
        response = service.files().list(
            q=f"trashed = false and '{parent_id}' in parents",
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, parents)",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return items


def scan_remote_tree(
    service: Any,
    root_id: str,
    *,
    ignore_matcher: IgnoreMatcher | None = None,
    progress_every: int = 500,
) -> tuple[dict[str, RemoteEntry], dict[str, RemoteEntry], tuple[RemoteEntry, ...]]:
    remote_files: dict[str, RemoteEntry] = {}
    remote_folders: dict[str, RemoteEntry] = {}
    remote_duplicates: list[RemoteEntry] = []
    scanned_count = 0

    def walk(folder_id: str, rel_prefix: str) -> None:
        nonlocal scanned_count
        for item in list_children(service, folder_id):
            scanned_count += 1
            if progress_every > 0 and scanned_count % progress_every == 0:
                print(f"Drive Scan : {scanned_count} remote items", flush=True)

            rel_path = normalize_rel_path(PurePosixPath(rel_prefix, item["name"]))
            entry = RemoteEntry(
                id=item["id"],
                rel_path=rel_path,
                name=item["name"],
                parent_id=(item.get("parents") or [None])[0],
                mime_type=item["mimeType"],
            )
            if entry.is_folder:
                if rel_path in remote_folders:
                    remote_duplicates.append(entry)
                    continue
                remote_folders[rel_path] = entry
                if ignore_matcher is not None and ignore_matcher.matches(
                    rel_path,
                    is_dir=True,
                ):
                    continue
                walk(entry.id, rel_path)
            else:
                if rel_path in remote_files:
                    remote_duplicates.append(entry)
                    continue
                remote_files[rel_path] = entry

    walk(root_id, "")
    return remote_files, remote_folders, tuple(remote_duplicates)


def ensure_remote_parent_folder(
    service: Any,
    root_id: str,
    folder_cache: dict[str, str],
    rel_dir: str,
) -> str:
    normalized_dir = normalize_rel_path(rel_dir)
    if not normalized_dir:
        return root_id
    if normalized_dir in folder_cache:
        return folder_cache[normalized_dir]

    parent_path = normalize_rel_path(PurePosixPath(normalized_dir).parent)
    parent_id = ensure_remote_parent_folder(service, root_id, folder_cache, parent_path)
    folder_name = PurePosixPath(normalized_dir).name
    folder_id = ensure_drive_folder(service, parent_id, folder_name)
    folder_cache[normalized_dir] = folder_id
    return folder_id


def apply_sync_plan(
    service: Any,
    service_factory: Callable[[], Any],
    root_id: str,
    plan: SyncPlan,
    *,
    delete_remote: bool = True,
) -> None:
    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive dependencies are missing. "
            "Install them with `pip install -r requirements.launch.txt`."
        ) from exc

    folder_cache = {
        "": root_id,
        **{
            entry.rel_path: entry.id
            for entry in plan.preserved_remote
            if entry.is_folder
        },
    }

    if plan.duplicate_deletes:
        run_parallel_tasks(
            plan.duplicate_deletes,
            desc="Deleting duplicates",
            unit="item",
            label_fn=lambda remote_entry: remote_entry.rel_path,
            service_factory=service_factory,
            worker_fn=lambda worker_service, remote_entry: worker_service.files()
            .delete(fileId=remote_entry.id)
            .execute(),
        )

    if delete_remote:
        run_parallel_tasks(
            plan.deletes,
            desc="Deleting remote",
            unit="item",
            label_fn=lambda remote_entry: remote_entry.rel_path,
            service_factory=service_factory,
            worker_fn=lambda worker_service, remote_entry: worker_service.files()
            .delete(fileId=remote_entry.id)
            .execute(),
        )
    elif plan.deletes:
        print(
            f"Delete     : skipped {len(plan.deletes)} remote-only items "
            "(use --keep-remote to preserve them)",
            flush=True,
        )

    prepare_remote_parent_folders(service, root_id, folder_cache, plan.uploads)
    upload_parent_ids = {
        local_file.rel_path: folder_cache[
            normalize_rel_path(PurePosixPath(local_file.rel_path).parent)
        ]
        for local_file in plan.uploads
    }

    def upload_local_file(worker_service: Any, local_file: LocalFile) -> None:
        media = MediaFileUpload(str(local_file.abs_path), resumable=False)
        worker_service.files().create(
            body={
                "name": PurePosixPath(local_file.rel_path).name,
                "parents": [upload_parent_ids[local_file.rel_path]],
            },
            media_body=media,
            fields="id",
        ).execute()

    def update_local_file(worker_service: Any, task: UpdateTask) -> None:
        media = MediaFileUpload(str(task.local_file.abs_path), resumable=False)
        worker_service.files().update(
            fileId=task.remote_entry.id,
            media_body=media,
            fields="id",
        ).execute()

    run_parallel_tasks(
        plan.updates,
        desc="Updating remote",
        unit="file",
        label_fn=lambda task: task.local_file.rel_path,
        service_factory=service_factory,
        worker_fn=update_local_file,
    )

    run_parallel_tasks(
        plan.uploads,
        desc="Uploading local",
        unit="file",
        label_fn=lambda local_file: local_file.rel_path,
        service_factory=service_factory,
        worker_fn=upload_local_file,
    )


def escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def prepare_remote_parent_folders(
    service: Any,
    root_id: str,
    folder_cache: dict[str, str],
    uploads: tuple[LocalFile, ...],
) -> None:
    parent_dirs = sorted(
        {
            normalize_rel_path(PurePosixPath(local_file.rel_path).parent)
            for local_file in uploads
        },
        key=lambda rel_path: (rel_path.count("/"), rel_path),
    )

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    print(f"Folder Prep: {len(parent_dirs)} parent folders", flush=True)
    iterator = (
        tqdm(parent_dirs, desc="Preparing folders", unit="folder")
        if tqdm is not None
        else parent_dirs
    )
    for index, rel_dir in enumerate(iterator, start=1):
        if tqdm is None or index == 1 or index % 25 == 0:
            print(f"Folder Prep: {index}/{len(parent_dirs)} {rel_dir}", flush=True)
        ensure_remote_parent_folder(service, root_id, folder_cache, rel_dir)


class ProgressTracker:
    def __init__(self, progress_bar: Any | None):
        self.progress_bar = progress_bar
        self._active_labels: dict[str, int] = {}
        self._lock = Lock()
        self._sequence = 0

    def start(self, label: str) -> None:
        with self._lock:
            self._sequence += 1
            self._active_labels[label] = self._sequence
            self._refresh_locked()

    def finish(self, label: str) -> None:
        with self._lock:
            self._active_labels.pop(label, None)
            self._refresh_locked()

    def _refresh_locked(self) -> None:
        if self.progress_bar is None:
            return

        if self._active_labels:
            current = max(self._active_labels, key=self._active_labels.__getitem__)
            self.progress_bar.set_postfix_str(
                shorten_progress_label(current),
                refresh=False,
            )
            return

        self.progress_bar.set_postfix_str("", refresh=False)


def shorten_progress_label(label: str, *, width: int = PROGRESS_LABEL_WIDTH) -> str:
    if len(label) <= width:
        return label
    return f"...{label[-(width - 3):]}"


def resolve_sync_workers(item_count: int) -> int:
    if item_count <= 0:
        return 0
    return min(DEFAULT_SYNC_WORKERS, item_count)


def run_parallel_tasks(
    items: tuple[T, ...],
    *,
    desc: str,
    unit: str,
    label_fn: Callable[[T], str],
    service_factory: Callable[[], Any],
    worker_fn: Callable[[Any, T], None],
) -> None:
    if not items:
        return

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    max_workers = resolve_sync_workers(len(items))
    thread_state = local()
    progress_bar = (
        tqdm(total=len(items), desc=desc, unit=unit)
        if tqdm is not None
        else None
    )
    tracker = ProgressTracker(progress_bar)

    def get_thread_service() -> Any:
        worker_service = getattr(thread_state, "service", None)
        if worker_service is None:
            worker_service = service_factory()
            thread_state.service = worker_service
        return worker_service

    def run_item(item: T) -> None:
        label = label_fn(item)
        tracker.start(label)
        try:
            worker_fn(get_thread_service(), item)
        finally:
            tracker.finish(label)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending = {executor.submit(run_item, item) for item in items}
            while pending:
                done, pending = wait(
                    pending,
                    timeout=0.1,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    future.result()
                    if progress_bar is not None:
                        progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()


def format_dest_path(dest_segments: list[str]) -> str:
    return f"MyDrive/{'/'.join(dest_segments)}"


def print_plan_summary(
    *,
    dest_segments: list[str],
    ignored_paths: tuple[str, ...],
    plan: SyncPlan,
    dry_run: bool,
    verbose: bool,
) -> None:
    print(f"Drive Path : {format_dest_path(dest_segments)}")
    print(f"Ignored    : {len(ignored_paths)}")
    print(f"Preserve   : {len(plan.preserved_remote)}")
    print(f"Upload     : {len(plan.uploads)}")
    print(f"Update     : {len(plan.updates)}")
    print(f"Deduplicate: {len(plan.duplicate_deletes)}")
    print(f"Delete     : {len(plan.deletes)}")
    if dry_run:
        print("Mode       : dry-run")

    if not verbose:
        return

    for local_file in plan.uploads:
        print(f"[UPLOAD] {local_file.rel_path}")
    for update_task in plan.updates:
        print(f"[UPDATE] {update_task.local_file.rel_path}")
    for remote_entry in plan.duplicate_deletes:
        print(f"[DEDUP DELETE] {remote_entry.rel_path}")
    for remote_entry in plan.deletes:
        print(f"[DELETE] {remote_entry.rel_path}")
    for remote_entry in plan.preserved_remote:
        print(f"[KEEP] {remote_entry.rel_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync the current repository to Google Drive for Colab execution.",
    )
    parser.add_argument(
        "--dest-path",
        required=True,
        help="Drive path under MyDrive, for example `Lab/GUARD_EXP_REPO`.",
    )
    parser.add_argument(
        "--client-secret",
        required=True,
        type=Path,
        help="Path to the Google OAuth client secret JSON file.",
    )
    parser.add_argument(
        "--token-path",
        type=Path,
        default=DEFAULT_TOKEN_PATH,
        help=(
            "Path to the cached OAuth token JSON file. "
            f"Defaults to `{DEFAULT_TOKEN_PATH}`."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root to sync. Defaults to the current project root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview uploads and deletes without changing Google Drive.",
    )
    parser.add_argument(
        "--keep-remote",
        action="store_true",
        help=(
            "Preserve remote-only Drive files that are not present locally. "
            "By default, remote-only files are deleted."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed file-level sync actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    client_secret_path = args.client_secret.expanduser().resolve()
    token_path = args.token_path.expanduser().resolve()
    dest_segments = parse_dest_path(args.dest_path)

    print(f"Repo Root  : {repo_root}", flush=True)
    print("Local Scan : scanning files", flush=True)
    ignore_matcher = IgnoreMatcher(load_ignore_patterns(repo_root))
    scan_result = collect_local_files(repo_root, ignore_matcher)
    print(f"Local Scan : {len(scan_result.files)} upload candidates", flush=True)

    print("Drive      : connecting", flush=True)
    service = get_drive_service(
        client_secret_path=client_secret_path,
        token_path=token_path,
    )

    print("Drive      : resolving destination", flush=True)
    root_id = resolve_dest_root(service, dest_segments, create=not args.dry_run)
    if root_id is None:
        plan = SyncPlan(
            uploads=tuple(scan_result.files[rel_path] for rel_path in sorted(scan_result.files)),
            updates=(),
            deletes=(),
            duplicate_deletes=(),
            preserved_remote=(),
        )
        print_plan_summary(
            dest_segments=dest_segments,
            ignored_paths=scan_result.ignored_paths,
            plan=plan,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        print("Note       : destination does not exist; dry-run skipped remote scan")
        return

    print("Drive      : scanning remote tree", flush=True)
    remote_files, remote_folders, remote_duplicates = scan_remote_tree(
        service,
        root_id,
        ignore_matcher=ignore_matcher,
    )
    print(
        f"Drive      : {len(remote_files)} files, {len(remote_folders)} folders, "
        f"{len(remote_duplicates)} duplicates found",
        flush=True,
    )
    plan = build_sync_plan(
        scan_result.files,
        remote_files,
        remote_folders,
        remote_duplicates,
        ignore_matcher,
    )

    print_plan_summary(
        dest_segments=dest_segments,
        ignored_paths=scan_result.ignored_paths,
        plan=plan,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    if args.dry_run:
        return

    service_factory = lambda: get_drive_service(
        client_secret_path=client_secret_path,
        token_path=token_path,
    )

    apply_sync_plan(
        service,
        service_factory,
        root_id,
        plan,
        delete_remote=not args.keep_remote,
    )
    print("Result     : sync completed")


def _matches_gitignore_patterns(rel_path: str, patterns: list[str]) -> bool:
    matched = False
    for pattern in patterns:
        raw_pattern = pattern.strip()
        if not raw_pattern:
            continue

        negated = raw_pattern.startswith("!")
        if negated:
            raw_pattern = raw_pattern[1:]

        if _matches_gitignore_pattern(rel_path, raw_pattern):
            matched = not negated

    return matched


def _matches_gitignore_pattern(rel_path: str, pattern: str) -> bool:
    path = normalize_rel_path(rel_path)
    if not path:
        return False

    raw_pattern = pattern.strip()
    if not raw_pattern:
        return False

    dir_only = raw_pattern.endswith("/")
    raw_pattern = raw_pattern.rstrip("/")
    anchored = raw_pattern.startswith("/")
    if anchored:
        raw_pattern = raw_pattern[1:]
    if not raw_pattern:
        return False

    parts = path.split("/")
    wildcard = any(char in raw_pattern for char in "*?[]")

    if anchored:
        return _match_path_candidate(path, raw_pattern, dir_only=dir_only)

    if "/" not in raw_pattern and not wildcard:
        if raw_pattern in parts:
            return True
        return parts[-1] == raw_pattern

    suffixes = ["/".join(parts[index:]) for index in range(len(parts))]
    candidates = [path, parts[-1], *suffixes]
    return any(
        _match_path_candidate(candidate, raw_pattern, dir_only=dir_only)
        for candidate in candidates
    )


def _match_path_candidate(candidate: str, pattern: str, *, dir_only: bool) -> bool:
    if candidate == pattern or candidate.startswith(f"{pattern}/"):
        return True
    if fnmatch(candidate, pattern):
        return True
    if not dir_only and fnmatch(PurePosixPath(candidate).name, pattern):
        return True
    return False


if __name__ == "__main__":
    main()
