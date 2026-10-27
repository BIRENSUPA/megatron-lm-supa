import glob
import os
import shutil
import site
import subprocess

import setuptools

# Set megatron open version
version = "0.19.0"
build_metadata = os.getenv('BUILD_METADATA', default='no_build_metedata')

def remove_files(pattern):
    files = glob.glob(pattern)
    for file in files:
        print(f"removing: {file}")
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)


names = ['megatron_supa*', 'apex*']
for site_pkg in site.getsitepackages():
    for name in names:
        path_pattern = f"{site_pkg}/{name}"
        remove_files(path_pattern)


def megatron_supa_version() -> str:
    """Megatron-LM-br version string

    Includes Git commit.

    """
    global version
    root_path = Path(__file__).resolve().parent
    try:
        output = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            cwd=root_path,
            check=True,
            universal_newlines=True,
        )
    except (subprocess.CalledProcessError, OSError):
        pass
    else:
        commit = output.stdout.strip()
        version += f"+{commit}"

    return version


def get_full_version() -> str:
    from pathlib import Path
    this_dir = Path(__file__).parent
    br_version = (this_dir / 'version.txt').read_text().replace("\n", "")
    br_release_label = br_version.split(".")[-1]
    public_version_separator = "." if br_release_label.startswith(("dev", "post")) else ""
    replace_str_release_label = public_version_separator + br_release_label

    global version
    version = version.replace('\n', '')
    full_version_specifiers = r"{}{}".format(version, replace_str_release_label)
    return full_version_specifiers


setuptools.setup(
    name="megatron_supa",
    version=get_full_version(),
    description="An adaptor for megatron on Biren GPU" + " --> build_metadata:{}".format(build_metadata),
    packages=setuptools.find_packages(include=['megatron_supa*']),
    # setuptools.find_packages(where='third_party/apex', include=['apex*']),
    package_dir={
        'megatron_supa': 'megatron_supa',
    },
    package_data={
        "megatron_supa.core": ["build/*.so"],
    },
    # install_package_data=True,
    include_package_data=True,
    license='Apache2',
    # license_file='./LICENSE',
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
)
