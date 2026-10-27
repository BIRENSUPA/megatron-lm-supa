import os
from pathlib import Path

import pytest
import torch
import torch.distributed

from tests.test_utils.python_scripts.download_unit_tests_dataset import download_and_extract_asset
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def pytest_addoption(parser):
    """
    Additional command-line arguments passed to pytest.
    For now:
        --experimental: Enable the Mcore experimental flag (DEFAULT: False)
    """
    parser.addoption(
        '--experimental',
        action='store_true',
        help="pass that argument to enable experimental flag during testing (DEFAULT: False)",
    )


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


@pytest.fixture(scope="session")
def tmp_path_dist_ckpt(tmp_path_factory) -> Path:
    """Common directory for saving the checkpoint.

    Can't use pytest `tmp_path_factory` directly because directory must be shared between processes.
    """

    tmp_dir = tmp_path_factory.mktemp('ignored', numbered=False)
    tmp_dir = tmp_dir.parent.parent / 'tmp_dist_ckpt'

    if Utils.rank == 0:
        with TempNamedDir(tmp_dir, sync=False):
            yield tmp_dir

    else:
        yield tmp_dir


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Ensure test data is available at /share/datasets by downloading if necessary."""
    data_path = Path("/share/datasets")

    # Check if data directory exists and has content
    if not data_path.exists() or not any(data_path.iterdir()):
        print("Test data not found at /share/datasets. Downloading...")

        try:
            # Download assets to /share/datasets
            download_and_extract_asset(data_path)

            print("Test data downloaded successfully.")

        except ImportError as e:
            print(f"Failed to import download function: {e}")
            # Don't fail the tests, just warn
        except Exception as e:
            print(f"Failed to download test data: {e}")
            # Don't fail the tests, just warn
    else:
        print("Test data already available at /share/datasets")


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables"""
    # Store the original environment variables before the test
    original_env = dict(os.environ)

    # Run the test
    yield

    # After the test, restore the original environment
    os.environ.clear()
    os.environ.update(original_env)
