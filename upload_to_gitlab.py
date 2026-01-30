#!/usr/bin/env python3
"""
Upload files to GitLab with Git LFS support.

This script provides a full-featured CLI for uploading files to GitLab,
supporting the flags-first syntax for specifying multiple file groups
with different LFS settings in a single commit.

Uses manual argv parsing for full control over the flags-first syntax,
importing the upload functionality from gitlab_to_hf_dataset.py.

Usage Examples:
    # Upload a single file to LFS
    python upload_to_gitlab.py --token TOKEN --project-id 454 \
        --lfs --path gpu_jobs/test/model.pt --file ./checkpoint.pt

    # Upload multiple files to a folder with LFS
    python upload_to_gitlab.py --token TOKEN --project-path ns/proj \
        --lfs --folder gpu_jobs/job_123/models/ --files model.pt checkpoint.pt

    # Upload inline content (no LFS)
    python upload_to_gitlab.py --token TOKEN --project-id 454 \
        --no-lfs --path gpu_jobs/job_123/response.yaml --content "state: completed"

    # Combined: LFS files + non-LFS content in single commit
    python upload_to_gitlab.py --token TOKEN --project-id 454 \
        --lfs --folder gpu_jobs/job_123/audio/ --files audio1.wav audio2.wav \
        --no-lfs --path gpu_jobs/job_123/response.yaml --content "state: completed"
"""

import os
import sys
from typing import List

from gitlab_to_hf_dataset import GitLabDatasetDownloader
from gitlab_jobs import DEFAULT_GITLAB_URL


def print_usage():
    """Print usage information."""
    print("""
Usage: upload_to_gitlab.py [connection options] [file specifications]

Connection Options:
  --token TOKEN           GitLab access token (or set GITLAB_TOKEN env var)
  --gitlab-url URL        GitLab server URL (default: https://git.genesisrnd.com)
  --project-id ID         Numeric project ID
  --project-path PATH     Project path like "namespace/project"
  --config-path PATH      Optional path to config.yaml for credentials
  --commit-message MSG    Optional commit message (auto-generated if not provided)
  --help, -h              Show this help message

File Specification Syntax:
  The file arguments use a flags-first syntax that allows multiple file groups
  with different LFS settings in a single commit.

  --lfs / --no-lfs        Set LFS mode for subsequent files (required before files)
  --folder <dir>          Set remote folder for subsequent --files
  --files <f1> <f2> ...   Upload local files to the folder (uses filename as remote name)
  --path <remote>         Set specific remote path for a single file
  --file <local>          Upload local file to the --path location
  --content <text>        Upload inline text content to the --path location

Examples:
  # Upload a single file to LFS
  upload_to_gitlab.py --token TOKEN --project-id 454 \\
      --lfs --path gpu_jobs/test/model.pt --file ./checkpoint.pt

  # Upload multiple files to a folder with LFS
  upload_to_gitlab.py --token TOKEN --project-path ns/proj \\
      --lfs --folder gpu_jobs/job_123/models/ --files model.pt checkpoint.pt

  # Upload inline content (no LFS)
  upload_to_gitlab.py --token TOKEN --project-id 454 \\
      --no-lfs --path gpu_jobs/job_123/response.yaml --content "state: completed"

  # Combined: LFS files + non-LFS content in single commit
  upload_to_gitlab.py --token TOKEN --project-id 454 \\
      --lfs --folder gpu_jobs/job_123/audio/ --files audio1.wav audio2.wav \\
      --no-lfs --path gpu_jobs/job_123/response.yaml --content "state: completed"
""")


def parse_args(argv: List[str]) -> tuple:
    """Parse all command line arguments.

    Args:
        argv: Command line arguments (sys.argv[1:])

    Returns:
        Tuple of (connection_options dict, file_specs list)
    """
    # Connection options
    options = {
        'token': None,
        'gitlab_url': DEFAULT_GITLAB_URL,
        'project_id': None,
        'project_path': None,
        'config_path': None,
        'commit_message': None,
    }

    # File specifications
    file_specs = []

    # Current file group settings
    use_lfs = None
    folder = None

    i = 0
    while i < len(argv):
        arg = argv[i]

        # Connection options (with = syntax support)
        if arg.startswith('--token='):
            options['token'] = arg.split('=', 1)[1]
            i += 1
        elif arg == '--token':
            if i + 1 >= len(argv):
                raise ValueError("--token requires a value")
            options['token'] = argv[i + 1]
            i += 2

        elif arg.startswith('--gitlab-url='):
            options['gitlab_url'] = arg.split('=', 1)[1]
            i += 1
        elif arg == '--gitlab-url':
            if i + 1 >= len(argv):
                raise ValueError("--gitlab-url requires a value")
            options['gitlab_url'] = argv[i + 1]
            i += 2

        elif arg.startswith('--project-id='):
            options['project_id'] = arg.split('=', 1)[1]
            i += 1
        elif arg == '--project-id':
            if i + 1 >= len(argv):
                raise ValueError("--project-id requires a value")
            options['project_id'] = argv[i + 1]
            i += 2

        elif arg.startswith('--project-path='):
            options['project_path'] = arg.split('=', 1)[1]
            i += 1
        elif arg == '--project-path':
            if i + 1 >= len(argv):
                raise ValueError("--project-path requires a value")
            options['project_path'] = argv[i + 1]
            i += 2

        elif arg.startswith('--config-path='):
            options['config_path'] = arg.split('=', 1)[1]
            i += 1
        elif arg == '--config-path':
            if i + 1 >= len(argv):
                raise ValueError("--config-path requires a value")
            options['config_path'] = argv[i + 1]
            i += 2

        elif arg.startswith('--commit-message='):
            options['commit_message'] = arg.split('=', 1)[1]
            i += 1
        elif arg == '--commit-message':
            if i + 1 >= len(argv):
                raise ValueError("--commit-message requires a value")
            options['commit_message'] = argv[i + 1]
            i += 2

        elif arg in ('--help', '-h'):
            print_usage()
            sys.exit(0)

        # LFS mode flags
        elif arg == '--lfs':
            use_lfs = True
            i += 1
        elif arg == '--no-lfs':
            use_lfs = False
            i += 1

        # Folder for multiple files
        elif arg.startswith('--folder='):
            folder = arg.split('=', 1)[1].rstrip('/')
            i += 1
        elif arg == '--folder':
            if i + 1 >= len(argv):
                raise ValueError("--folder requires a path argument")
            folder = argv[i + 1].rstrip('/')
            i += 2

        # Multiple files to folder
        elif arg == '--files':
            if use_lfs is None:
                raise ValueError("--files requires --lfs or --no-lfs to be specified first")
            if folder is None:
                raise ValueError("--files requires --folder to be specified first")

            # Collect all file arguments until next flag
            i += 1
            while i < len(argv) and not argv[i].startswith('--'):
                local_path = argv[i]
                filename = os.path.basename(local_path)
                remote_path = f"{folder}/{filename}"
                file_specs.append({
                    'local_path': local_path,
                    'remote_path': remote_path,
                    'lfs': use_lfs
                })
                i += 1

        # Single file path specification
        elif arg.startswith('--path='):
            remote_path = arg.split('=', 1)[1]
            i += 1

            # Next should be --content or --file
            if i >= len(argv):
                raise ValueError("--path must be followed by --content or --file")

            next_arg = argv[i]
            if next_arg.startswith('--content='):
                content = next_arg.split('=', 1)[1]
                i += 1
                if use_lfs is None:
                    raise ValueError("--path requires --lfs or --no-lfs to be specified first")
                file_specs.append({
                    'content': content,
                    'remote_path': remote_path,
                    'lfs': use_lfs
                })
            elif next_arg == '--content':
                if i + 1 >= len(argv):
                    raise ValueError("--content requires a content argument")
                content = argv[i + 1]
                i += 2
                if use_lfs is None:
                    raise ValueError("--path requires --lfs or --no-lfs to be specified first")
                file_specs.append({
                    'content': content,
                    'remote_path': remote_path,
                    'lfs': use_lfs
                })
            elif next_arg.startswith('--file='):
                local_path = next_arg.split('=', 1)[1]
                i += 1
                if use_lfs is None:
                    raise ValueError("--path requires --lfs or --no-lfs to be specified first")
                file_specs.append({
                    'local_path': local_path,
                    'remote_path': remote_path,
                    'lfs': use_lfs
                })
            elif next_arg == '--file':
                if i + 1 >= len(argv):
                    raise ValueError("--file requires a local file path argument")
                local_path = argv[i + 1]
                i += 2
                if use_lfs is None:
                    raise ValueError("--path requires --lfs or --no-lfs to be specified first")
                file_specs.append({
                    'local_path': local_path,
                    'remote_path': remote_path,
                    'lfs': use_lfs
                })
            else:
                raise ValueError("--path must be followed by --content or --file")

        elif arg == '--path':
            if i + 1 >= len(argv):
                raise ValueError("--path requires a remote path argument")
            remote_path = argv[i + 1]
            i += 2

            # Next should be --content or --file
            if i >= len(argv):
                raise ValueError("--path must be followed by --content or --file")

            next_arg = argv[i]
            if next_arg.startswith('--content='):
                content = next_arg.split('=', 1)[1]
                i += 1
                if use_lfs is None:
                    raise ValueError("--path requires --lfs or --no-lfs to be specified first")
                file_specs.append({
                    'content': content,
                    'remote_path': remote_path,
                    'lfs': use_lfs
                })
            elif next_arg == '--content':
                if i + 1 >= len(argv):
                    raise ValueError("--content requires a content argument")
                content = argv[i + 1]
                i += 2
                if use_lfs is None:
                    raise ValueError("--path requires --lfs or --no-lfs to be specified first")
                file_specs.append({
                    'content': content,
                    'remote_path': remote_path,
                    'lfs': use_lfs
                })
            elif next_arg.startswith('--file='):
                local_path = next_arg.split('=', 1)[1]
                i += 1
                if use_lfs is None:
                    raise ValueError("--path requires --lfs or --no-lfs to be specified first")
                file_specs.append({
                    'local_path': local_path,
                    'remote_path': remote_path,
                    'lfs': use_lfs
                })
            elif next_arg == '--file':
                if i + 1 >= len(argv):
                    raise ValueError("--file requires a local file path argument")
                local_path = argv[i + 1]
                i += 2
                if use_lfs is None:
                    raise ValueError("--path requires --lfs or --no-lfs to be specified first")
                file_specs.append({
                    'local_path': local_path,
                    'remote_path': remote_path,
                    'lfs': use_lfs
                })
            else:
                raise ValueError("--path must be followed by --content or --file")

        elif arg == '--content':
            raise ValueError("--content must be preceded by --path")
        elif arg == '--file':
            raise ValueError("--file must be preceded by --path (use --files with --folder for multiple files)")
        else:
            raise ValueError(f"Unknown argument: {arg}")

    return options, file_specs


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    try:
        options, file_specs = parse_args(sys.argv[1:])
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Get token from parameter or environment
    access_token = options['token'] or os.environ.get('GITLAB_TOKEN')
    if not access_token:
        print("ERROR: No GitLab token provided. Use --token or set GITLAB_TOKEN env var.")
        sys.exit(1)

    # Require project identification
    if not options['project_id'] and not options['project_path']:
        print("ERROR: Must provide either --project-id or --project-path")
        sys.exit(1)

    if not file_specs:
        print("ERROR: No files specified for upload")
        print("Use --lfs --folder <dir> --files <file1> ... or --no-lfs --path <path> --content <text>")
        sys.exit(1)

    print("=" * 60)
    print("GitLab LFS Upload")
    print("=" * 60)
    print()

    print(f"Files to upload: {len(file_specs)}")
    for spec in file_specs:
        lfs_str = "LFS" if spec.get('lfs') else "git"
        if 'local_path' in spec:
            print(f"  [{lfs_str}] {spec['local_path']} -> {spec['remote_path']}")
        else:
            content_preview = spec.get('content', '')[:50]
            if len(spec.get('content', '')) > 50:
                content_preview += '...'
            print(f"  [{lfs_str}] <content> -> {spec['remote_path']}")
    print()

    # Create uploader instance
    uploader = GitLabDatasetDownloader(
        config_path=options['config_path'],
        gitlab_url=options['gitlab_url'],
        access_token=access_token,
        project_id=options['project_id'],
        project_path=options['project_path'],
    )

    print(f"Server: {uploader.server_url}")
    print(f"Project: {uploader.project_path or uploader.project_id_number}")
    print()

    # Perform upload
    result = uploader.upload_batch(file_specs, commit_message=options['commit_message'])

    # Print results
    if result['success']:
        print("✓ Upload successful!")
        print(f"  Commit SHA: {result['commit_sha']}")
        print(f"  Files uploaded: {len(result['files_uploaded'])}")
        for f in result['files_uploaded']:
            lfs_str = "LFS" if f.get('lfs') else "git"
            print(f"    [{lfs_str}] {f['remote_path']} ({f.get('size', 0)} bytes)")
        sys.exit(0)
    else:
        print("✗ Upload failed!")
        if result.get('error_message'):
            print(f"  Error: {result['error_message']}")
        if result.get('files_failed'):
            print(f"  Failed files: {len(result['files_failed'])}")
            for f in result['files_failed']:
                print(f"    {f['remote_path']}: {f['error']}")
        if result.get('files_uploaded'):
            print(f"  Partially uploaded: {len(result['files_uploaded'])} files")
        sys.exit(1)


if __name__ == "__main__":
    main()
