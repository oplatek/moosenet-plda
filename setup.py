import os
from pathlib import Path
from subprocess import DEVNULL, PIPE, run

from setuptools import find_packages, setup

# setup.py heavily inspired in lhotse-speech/lhotse project (Apache License 2.0)
project_root = Path(__file__).parent
MAJOR_VERSION = 0
MINOR_VERSION = 0
PATCH_VERSION = 1
IS_DEV_VERSION = True  # False = public release, True = otherwise


def discover_version() -> str:
    """
    When development version is detected, it queries git for the commit hash
    to append it as a local version identifier.
    """

    version = f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}"
    if not IS_DEV_VERSION:
        # This is a PyPI public release -- return a clean version string.
        return version

    version = version + ".dev"

    # This is not a PyPI release -- try to read the git commit
    try:
        git_commit = (
            run(
                ["git", "rev-parse", "--short", "HEAD"],
                check=True,
                stdout=PIPE,
                stderr=DEVNULL,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
        dirty_commit = (
            len(
                run(
                    ["git", "diff", "--shortstat"],
                    check=True,
                    stdout=PIPE,
                    stderr=DEVNULL,
                )
                .stdout.decode()
                .rstrip("\n")
                .strip()
            )
            > 0
        )
        git_commit = git_commit + ".dirty" if dirty_commit else git_commit + ".clean"
        source_version = f"+git.{git_commit}"
    except Exception:
        source_version = ".unknownsource"
    # See the format:
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#local-version-identifiers
    version = version + source_version

    return version


def mark_version(version: str) -> None:
    (project_root / "moosenet" / "version.py").write_text(f'__version__ = "{version}"')


VERSION = discover_version()
mark_version(VERSION)


install_requires = [
    "pytorch-lightning==1.5.10",
    "lhotse @ git+https://github.com/lhotse-speech/lhotse@08a613a06d258ab72679f0d2e952ec668afbe185#egg=lhotse",
    "plda @ git+https://github.com/oplatek/plda@abde92c48916738c4e90cc488e360723ed921dc8#egg=plda",
    "g2p_en==2.1.0",
    "torch_optimizer==0.3.0",
    "fastdtw==0.3.4",
    "torchmetrics==0.7.0",
    "torchmetrics[audio]",
    "torch==1.10.1",
    "torchaudio==0.10.1",
    "wandb==0.12.10",
    "click>=8.0.3",
    "tqdm>=4.62.3",
    "librosa==0.8.1",
    "torchlibrosa==0.0.9",
    "plotly==5.6.0",
]


docs_require = []  # no docs ATM
tests_require = [
    "pytest==5.4.3",
    "flake8==3.8.3",
]

dev_requires = sorted(
    docs_require
    + tests_require
    + [
        "black==22.1.0",
        "jupyterlab",
        "matplotlib",
        "isort",
        "ipdb",
        "py-spy",
        "pandas",
        "exp-notifier",
        "tensorboardX",
    ]
)
all_requires = sorted(dev_requires)

setup(
    name="moosenet-plda",
    version=VERSION,
    python_requires=">=3.8.0",
    description="Attempt to predict MOS (Mean Opinion Score) or Moose behaviour. Whichever is easier.",
    author="Ondrej Platek (oplatek)",
    author_email="ondrej.platek@seznam.cz or oplatek@ufal.mff.cuni.cz",
    long_description=(project_root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="Apache-2.0 License",
    packages=find_packages(exclude=["test", "test.*"]),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "moosenet=moosenet.bin.cli:moosenet",
        ]
    },
    install_requires=install_requires,
    extras_require={
        "docs": docs_require,
        "tests": tests_require,
        "dev": dev_requires,
        "all": all_requires,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
)
