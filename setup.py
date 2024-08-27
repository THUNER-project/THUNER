"""
Package setup methods.
"""

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from pathlib import Path
import glob
import json
import os


DOCLINES = __doc__.split("\n")


def create_user_config(output_directory=Path.home() / "THOR_output"):
    # Determine the OS-specific path
    if os.name == "nt":  # Windows
        config_path = Path(os.getenv("LOCALAPPDATA")) / "THOR" / "config.json"
    elif os.name == "posix":
        if "HOME" in os.environ:  # Linux/macOS
            config_path = Path.home() / ".config" / "THOR" / "config.json"
        else:  # Fallback for other POSIX systems
            config_path = Path("/etc") / "THOR" / "config.json"
    else:
        raise Exception("Unsupported operating system.")

    # Ensure the config directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a new config.json with initial settings
    with open(config_path, "w") as config_file:
        json.dump({"outputs_directory": str(output_directory)}, config_file)
        print(f"Created new configuration file at {config_path}")

    return str(config_path)


def post_setup():
    """Allow user to specify the default output directory."""
    if os.isatty(0):  # Check if running in an interactive terminal
        output_dir = input(
            "Please specify the default output directory. "
            f"Leave blank for {Path.home() / 'THOR_output'}: "
        )
    else:
        output_dir = ""

    if output_dir == "":
        output_dir = Path.home() / "THOR_output"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except FileNotFoundError:
        print("Invalid directory. Using default.")
        output_dir = Path.home() / "THOR_output"
        output_dir.mkdir(parents=True, exist_ok=True)

    create_user_config(output_dir)


class CustomInstall(install):
    def run(self):
        install.run(self)
        post_setup()


class CustomDevelop(develop):
    def run(self):
        develop.run(self)
        post_setup()


def read(pkg_name):
    init_fname = Path(__file__).parent / pkg_name / "__init__.py"
    with open(init_fname, "r") as fp:
        return fp.read()


def get_version(pkg_name):
    for line in read(pkg_name).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


def get_requirements(requirements_filename):
    requirements_file = Path(__file__).parent / requirements_filename
    assert requirements_file.exists()
    with open(requirements_file) as f:
        requirements = [
            line.strip() for line in f.readlines() if not line.startswith("#")
        ]
    return requirements


def get_packages(package_name):
    package = Path(package_name)
    packages = [
        str(path.parent).replace("/", ".") for path in package.rglob("__init__.py")
    ]
    return packages


# See classifiers list at: https://pypi.org/classifiers/
CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Environment :: Console",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
]

PACKAGE_NAME = "thor"
AUTHORS = "Ewan Short, Mark Picel, Bhupendra Raut"
MAINTAINER = "Ewan Short"
MAINTAINER_EMAIL = "ewan.short@unimelb.edu.au"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
URL = "https://github.com/THOR-proj/THOR.git"
DOWNLOAD_URL = "https://github.com/THOR-proj/THOR.git"
LICENSE = "BSD"
PLATFORMS = ["Linux", "Mac OS-X", "Unix"]
MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
VERSION = "%d.%d.%d" % (MAJOR, MINOR, MICRO)
SCRIPTS = glob.glob("scripts/*")


setup(
    name=PACKAGE_NAME,
    version=get_version(PACKAGE_NAME),
    description=("Thunderstorm hierachical object reconnoitrer"),
    url="http://github.com/THOR-proj/THOR",
    classifiers=CLASSIFIERS,
    author=[
        "Ewan Short",
    ],
    author_email=[
        "ewan.short@unimelb.edu.au",
    ],
    license="BSD-3-Clause License",
    packages=get_packages(PACKAGE_NAME),
    install_requires=get_requirements("requirements.txt"),
    # test_requires=["pytest"],
    zip_safe=False,
    cmdclass={
        "install": CustomInstall,
        "develop": CustomDevelop,
    },
)
