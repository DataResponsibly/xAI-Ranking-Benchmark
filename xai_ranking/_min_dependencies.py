"""All minimum dependencies."""

import argparse

# The values are (version_spec, comma separated tags)
dependent_packages = {
    "numpy": ("1.20.0", "install"),
    "pandas": ("1.3.5", "install"),
    "scipy": ("1.14.1", "install"),
    "scikit-learn": ("1.2.0", "install"),
    "pytest-cov": ("3.0.0", "tests"),
    "flake8": ("3.8.2", "tests"),
    "black": ("22.3", "tests"),
    "pylint": ("2.12.2", "tests"),
    "mypy": ("1.6.1", "tests"),
    "sphinx": ("4.2.0", "docs"),
    "statsmodels": ("0.14.4", "optional"),
    "lime": ("0.2.0.1", "optional"),
    "shap": ("0.46.0", "optional"),
    "xai-sharp": ("0.1a1", "optional"),
    # dev
    # "coverage": ("", "tests"),
    # "click": ("", "tests"),
    # nutrition labels
    # "matplotlib" : ("", "install"),
    # "seaborn" : ("", "install"),
    # L2R
    # "lightgbm" : ("", "install"),
    # general?
    # "ml-research" : ("", "install"),
    # dataset module
    # "openpyxl" : ("", "install"),
}

# create inverse mapping for setuptools
tag_to_packages: dict = {
    extra: [] for extra in ["install", "optional", "docs", "examples", "tests", "all"]
}
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append("{}>={}".format(package, min_version))
    tag_to_packages["all"].append("{}>={}".format(package, min_version))

# Used by CI to get the min dependencies
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies for a package")

    parser.add_argument("package", choices=dependent_packages)
    args = parser.parse_args()
    min_version = dependent_packages[args.package][0]
    print(min_version)
