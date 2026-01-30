# GitLab GPU Job Scanner

A Python CLI tool for scanning GitLab projects to find available GPU jobs defined in manifest files.

## Features

- Scans all accessible GitLab projects for GPU job manifests
- Identifies available (unclaimed, non-canceled) jobs
- Supports retry logic with exponential backoff for network resilience
- Provides detailed error reporting per project
- Outputs results as JSON for easy integration with other tools

## Installation

### Dependencies

Install the required Python packages:

```bash
pip install python-gitlab fire pyyaml
```

Or using the project's pyproject.toml if available:

```bash
pip install -e .
```

## Usage

### Basic Usage

List all available GPU jobs:

```bash
python gitlab_jobs.py list_available_jobs --token=YOUR_GITLAB_TOKEN
```

### Using Environment Variable for Token

Set the token as an environment variable:

```bash
export GITLAB_TOKEN=YOUR_GITLAB_TOKEN
python gitlab_jobs.py list_available_jobs
```

### Custom GitLab Server

Specify a custom GitLab server URL:

```bash
python gitlab_jobs.py list_available_jobs \
  --token=YOUR_TOKEN \
  --gitlab-url=https://custom.gitlab.com
```

### Verbose Output

Enable verbose logging to see scan progress:

```bash
python gitlab_jobs.py list_available_jobs \
  --token=YOUR_TOKEN \
  --verbose
```

### Quiet Mode

Suppress all output except the final JSON result:

```bash
python gitlab_jobs.py list_available_jobs \
  --token=YOUR_TOKEN \
  --quiet
```

### Custom Retry Configuration

Adjust the maximum number of retry attempts:

```bash
python gitlab_jobs.py list_available_jobs \
  --token=YOUR_TOKEN \
  --max-retries=5
```

### Scan All Projects (Raw Data)

Get the complete scan data for all projects (including errors):

```bash
python gitlab_jobs.py scan_all_projects --token=YOUR_TOKEN
```

### Create Files in Repository

Create a new file in a GitLab repository with direct content:

```bash
python gitlab_jobs.py create_file \
  --token=YOUR_TOKEN \
  --project-id=123 \
  --file-path=test.txt \
  --content="Hello, World!"
```

Create a file from a local file:

```bash
python gitlab_jobs.py create_file \
  --token=YOUR_TOKEN \
  --project-id=123 \
  --file-path=config.yaml \
  --file-source=./local-config.yaml
```

With a custom commit message:

```bash
python gitlab_jobs.py create_file \
  --token=YOUR_TOKEN \
  --project-id=123 \
  --file-path=README.md \
  --content="# My Project" \
  --commit-message="Initial README"
```

**Note**: The file will be created on the project's default branch. If the file already exists, a `GitLabFileExistsError` will be raised.

## Output Format

### list_available_jobs

Returns a JSON array of available jobs:

```json
[
  {
    "job_id": "job_001",
    "project_id": 123,
    "project_name": "My Project",
    "project_url": "https://git.genesisrnd.com/namespace/project/-/blob/main/gpu_jobs/manifest.yaml",
    "job_data": {
      "job_id": "job_001",
      "job_type": "training",
      "mode": "finetune",
      "model": "stable-tts",
      "epochs": 100,
      "canceled": false
    }
  }
]
```

### scan_all_projects

Returns detailed data for all scanned projects:

```json
[
  {
    "project_id": 123,
    "project_name": "My Project",
    "project_path": "namespace/project",
    "default_branch": "main",
    "manifest_url": "https://git.genesisrnd.com/namespace/project/-/blob/main/gpu_jobs/manifest.yaml",
    "manifest_data": {
      "version": 1,
      "jobs": [...]
    },
    "jobs": [
      {
        "job_id": "job_001",
        "is_claimed": false,
        "canceled": false,
        ...
      }
    ],
    "error": null
  }
]
```

## How It Works

### Job Detection

1. **Project Scanning**: Iterates through all GitLab projects accessible to the provided token
2. **Manifest Discovery**: Looks for `gpu_jobs/manifest.yaml` in each project's default branch
3. **Manifest Validation**: Validates manifest version (must be ≤ 1)
4. **Job Status Check**: For each job, checks if it's been claimed by looking for `gpu_jobs/job_<job_id>/response.yaml`
5. **Filtering**: Returns only jobs where:
   - `canceled` is not `true`
   - No `response.yaml` file exists (not claimed)

### Error Handling

The script implements robust error handling:

- **Authentication Errors**: Fails immediately with clear error message
- **Network Errors**: Retries with exponential backoff (configurable)
- **Per-Project Errors**: Captured in project data structure, doesn't stop scanning
- **Manifest Errors**: Reported per-project, allows other projects to succeed

### Retry Logic

Network and server errors are automatically retried with exponential backoff:

- Initial delay: 1 second
- Backoff multiplier: 2x
- Maximum delay: 60 seconds
- Default max retries: 10 (configurable)

Errors that are NOT retried:
- Authentication failures (401/403)
- Client errors (400-499)
- Missing resources (404)

## Integration with Other Scripts

### Python Integration

```python
from gitlab_jobs import GitLabJobScanner

# Create scanner
scanner = GitLabJobScanner(
    token="YOUR_TOKEN",
    gitlab_url="https://git.genesisrnd.com",
    verbose=True
)

# Get available jobs
jobs = scanner.list_available_jobs()

# Process jobs
for job in jobs:
    print(f"Job {job['job_id']} in project {job['project_name']}")
    print(f"  View manifest: {job['project_url']}")
```

### Shell Script Integration

```bash
#!/bin/bash

# Get available jobs as JSON
JOBS=$(python gitlab_jobs.py list_available_jobs --token=$GITLAB_TOKEN --quiet)

# Count available jobs
JOB_COUNT=$(echo "$JOBS" | jq 'length')
echo "Found $JOB_COUNT available jobs"

# Process each job
echo "$JOBS" | jq -c '.[]' | while read job; do
    JOB_ID=$(echo "$job" | jq -r '.job_id')
    PROJECT=$(echo "$job" | jq -r '.project_name')
    echo "Processing job $JOB_ID from $PROJECT"
done
```

## Manifest File Format

The script expects manifest files at `gpu_jobs/manifest.yaml` with this structure:

```yaml
version: 1
jobs:
  - job_id: "job_001"
    job_type: "training"
    mode: "finetune"
    model: "stable-tts"
    epochs: 100
    inference: false
    voice_reference: "speaker_001"
    timeout: 3600
    canceled: false
  - job_id: "job_002"
    ...
```

### Job Claiming

A job is considered "claimed" when a file exists at:
```
gpu_jobs/job_<job_id>/response.yaml
```

## Troubleshooting

### No Token Error

```
NoTokenProvidedError: No GitLab token provided
```

**Solution**: Provide token via `--token` parameter or `GITLAB_TOKEN` environment variable

### Authentication Failed

```
GitLabAuthenticationError: Authentication failed
```

**Solution**: Verify your token is valid and has appropriate permissions

### Connection Error

```
GitLabConnectionError: Unable to connect to GitLab server
```

**Solution**: 
- Check your internet connection
- Verify the GitLab URL is correct
- Check if the GitLab server is accessible

### No Jobs Found

If no jobs are returned:
- Verify projects have `gpu_jobs/manifest.yaml` files
- Check manifest version is ≤ 1
- Ensure jobs are not canceled or already claimed
- Use `--verbose` to see scan progress

## License

See project LICENSE file.