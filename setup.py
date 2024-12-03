import os
from pathlib import Path
from setuptools import find_packages, setup


def get_long_description() -> str:
    CURRENT_DIR = Path(__file__).parent
    return (CURRENT_DIR / "README.md").read_text(encoding="utf8")


import xai_ranking._min_dependencies as min_deps  # noqa

ver_file = os.path.join("xai_ranking", "_version.py")
with open(ver_file) as f:
    exec(f.read())

VERSION = __version__  # noqa
SHORT_DESCRIPTION = "Ranking Explainability Benchmark"
LICENSE = "MIT"
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    # "Intended Audience :: Developers",
    # "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    # "Operating System :: Microsoft :: Windows",
    # "Operating System :: POSIX",
    # "Operating System :: Unix",
    # "Operating System :: MacOS",
    # "Programming Language :: Python :: 3.9",
    # "Programming Language :: Python :: 3.10",
    # "Programming Language :: Python :: 3.11",
    # "Programming Language :: Python :: 3.12",
]
INSTALL_REQUIRES = (min_deps.tag_to_packages["metrics"],)
EXTRAS_REQUIRE = {
    key: value for key, value in min_deps.tag_to_packages.items() if key != "metrics"
}

setup(
    name="xai-ranking",
    version=VERSION,
    description=SHORT_DESCRIPTION,
    author="dataresponsibly",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
