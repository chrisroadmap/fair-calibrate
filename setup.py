import pathlib

from setuptools import find_packages, setup

AUTHORS = [
    ("Chris Smith", "c.j.smith1@leeds.ac.uk"),
]

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="fair_calibrate",
    version="1.6.0",
    description="Calibration mechanism for fair",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chrisroadmap/fair-calibrate",
    author=", ".join([author[0] for author in AUTHORS]),
    author_email=", ".join([author[1] for author in AUTHORS]),
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8, <4",
)
