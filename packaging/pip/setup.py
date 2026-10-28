# Builds the wheel from the staged directory build_wheel.py prepares (see pyproject.toml). Everything that varies
# by build comes from wheel.json next to this file: the version and the install requirements.
import json
import os

from setuptools import find_namespace_packages, setup

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wheel.json"), encoding="utf-8") as f:
    meta = json.load(f)

setup(
    version=meta["version"],
    install_requires=meta["requires"],
    # The tree under lernex/_bin is data, but its directories are packages to setuptools (no __init__.py), so
    # listing them all keeps the build quiet and ships every file once.
    packages=find_namespace_packages(include=["lernex", "lernex.*"]),
    package_data={"lernex._bin": ["*", "*/*", "*/*/*", "*/*/*/*", "*/*/*/*/*"]},
)
