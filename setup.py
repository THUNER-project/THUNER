"""
This code is from the python documentation and is
designed to read in the version number.
See: https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
"""

from setuptools import setup
from pathlib import Path
import glob

DOCLINES = __doc__.split("\n")


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
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
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
    test_requires=["pytest"],
    zip_safe=False,
)
