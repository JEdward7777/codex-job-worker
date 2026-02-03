#!/usr/bin/env python3
"""
Dry run test script for codex-job-worker.
Lists available jobs from GitLab to verify connectivity.
"""

import os
import sys

# Add the worker directory to path
sys.path.insert(0, os.path.expanduser("~/codex-job-worker"))

from gitlab_jobs import GitLabJobScanner

def main():
    print("Initializing GitLab scanner...")
    scanner = GitLabJobScanner(verbose=True)

    print("\nScanning for available jobs...")
    jobs = scanner.list_available_jobs()

    print("\n" + "=" * 60)
    print(f"Found {len(jobs)} available jobs:")
    print("=" * 60)

    for job in jobs:
        print(f"\nJob ID: {job.get('job_id')}")
        print(f"  Project: {job.get('project_id')}")
        print(f"  Type: {job.get('job_type')}")
        print(f"  Mode: {job.get('mode')}")
        print(f"  Claimed: {job.get('is_claimed')}")

    if not jobs:
        print("\nNo jobs available (this is normal if no jobs have been created)")
        print("The important thing is that GitLab connectivity works!")

    print("\n" + "=" * 60)
    print("DRY RUN COMPLETE - GitLab connectivity verified!")
    print("=" * 60)

if __name__ == "__main__":
    main()
