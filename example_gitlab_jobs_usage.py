#!/usr/bin/env python3
"""
Example usage of the GitLab Job Scanner

This demonstrates how to use the gitlab_jobs module programmatically.
"""

import os
import json
import traceback
from gitlab_jobs import GitLabJobScanner


def example_basic_usage():
    """Basic example: List available jobs."""
    print("=" * 60)
    print("Example 1: Basic Usage - List Available Jobs")
    print("=" * 60)

    # Get token from environment
    token = os.environ.get('GITLAB_TOKEN')
    if not token:
        print("ERROR: Please set GITLAB_TOKEN environment variable")
        return

    # Create scanner
    scanner = GitLabJobScanner(
        token=token,
        gitlab_url="https://git.genesisrnd.com",
        verbose=True
    )

    # Get available jobs
    jobs = scanner.list_available_jobs()

    # Display results
    print("\n" + "=" * 60)
    print(f"Found {len(jobs)} available jobs:")
    print("=" * 60)

    for i, job in enumerate(jobs, 1):
        print(f"\nJob {i}:")
        print(f"  Job ID: {job['job_id']}")
        print(f"  Project: {job['project_name']} (ID: {job['project_id']})")
        print(f"  Manifest URL: {job['project_url']}")
        print(f"  Job Type: {job['job_data'].get('job_type', 'N/A')}")
        print(f"  Model: {job['job_data'].get('model', 'N/A')}")


def example_scan_all_projects():
    """Example: Get detailed scan data for all projects."""
    print("\n" + "=" * 60)
    print("Example 2: Scan All Projects (Detailed Data)")
    print("=" * 60)

    token = os.environ.get('GITLAB_TOKEN')
    if not token:
        print("ERROR: Please set GITLAB_TOKEN environment variable")
        return

    scanner = GitLabJobScanner(
        token=token,
        gitlab_url="https://git.genesisrnd.com",
        verbose=False  # Quiet mode for cleaner output
    )

    # Get all project data
    all_projects = scanner.scan_all_projects()

    # Analyze results
    total_projects = len(all_projects)
    projects_with_manifests = sum(1 for p in all_projects if p['manifest_data'])
    projects_with_errors = sum(1 for p in all_projects if p['error'])
    total_jobs = sum(len(p['jobs']) for p in all_projects if p['jobs'])

    print("\nScan Summary:")
    print(f"  Total projects scanned: {total_projects}")
    print(f"  Projects with manifests: {projects_with_manifests}")
    print(f"  Projects with errors: {projects_with_errors}")
    print(f"  Total jobs found: {total_jobs}")

    # Show projects with errors
    if projects_with_errors > 0:
        print("\nProjects with errors:")
        for project in all_projects:
            if project['error']:
                print(f"  - {project['project_name']}: {project['error']}")


def example_filter_jobs():
    """Example: Filter jobs by specific criteria."""
    print("\n" + "=" * 60)
    print("Example 3: Filter Jobs by Criteria")
    print("=" * 60)

    token = os.environ.get('GITLAB_TOKEN')
    if not token:
        print("ERROR: Please set GITLAB_TOKEN environment variable")
        return

    scanner = GitLabJobScanner(
        token=token,
        gitlab_url="https://git.genesisrnd.com",
        verbose=False
    )

    # Get available jobs
    jobs = scanner.list_available_jobs()

    # Filter by job type
    training_jobs = [j for j in jobs if j['job_data'].get('job_type') == 'training']
    inference_jobs = [j for j in jobs if j['job_data'].get('inference') is True]

    print("\nJob Breakdown:")
    print(f"  Total available jobs: {len(jobs)}")
    print(f"  Training jobs: {len(training_jobs)}")
    print(f"  Inference jobs: {len(inference_jobs)}")

    # Show training jobs
    if training_jobs:
        print("\nTraining Jobs:")
        for job in training_jobs:
            print(f"  - {job['job_id']} in {job['project_name']}")
            print(f"    Model: {job['job_data'].get('model', 'N/A')}")
            print(f"    Epochs: {job['job_data'].get('epochs', 'N/A')}")


def example_json_output():
    """Example: Output as JSON for integration with other tools."""
    print("\n" + "=" * 60)
    print("Example 4: JSON Output for Integration")
    print("=" * 60)

    token = os.environ.get('GITLAB_TOKEN')
    if not token:
        print("ERROR: Please set GITLAB_TOKEN environment variable")
        return

    scanner = GitLabJobScanner(
        token=token,
        gitlab_url="https://git.genesisrnd.com",
        quiet=True  # Suppress all logging
    )

    # Get available jobs
    jobs = scanner.list_available_jobs()

    # Output as formatted JSON
    print(json.dumps(jobs, indent=2))


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("GitLab Job Scanner - Usage Examples")
    print("=" * 60)

    # Check for token
    if not os.environ.get('GITLAB_TOKEN'):
        print("\nERROR: GITLAB_TOKEN environment variable not set!")
        print("\nPlease set it before running examples:")
        print("  export GITLAB_TOKEN=your_token_here")
        print("\nThen run:")
        print("  python example_gitlab_jobs_usage.py")
        return

    try:
        # Run examples
        example_basic_usage()
        example_scan_all_projects()
        example_filter_jobs()
        example_json_output()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()