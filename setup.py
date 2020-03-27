import os
from setuptools import find_packages
from setuptools import setup


with open(os.path.join(os.path.dirname(__file__), "README.md"),
          encoding='utf-8') as readme:
    long_description = readme.read()

setup(name="quasildr",
      version="0.2.1",
      long_description=long_description,
      long_description_content_type='text/markdown',
      description=("quasilinear representation methods for single-cell"
                   "omics data"),
      packages=find_packages(),
      url="https://github.com/jzthree/quasildr",
      download_url="https://github.com/jzthree/quasildr/archive/v0.2.1.tar.gz",
      package_data={
      },
      scripts=[
        "run_graphdr.py",
        "run_structdr.py",
      ],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
      ],
      install_requires=[
        "docopt",
        "multiprocess",
        "numpy",
        "pandas",
        "plotly",
        "scikit-learn",
        "scipy",
        "seaborn",
        "statsmodels",
        "plotnine",
        "pynndescent",
        "nmslib"
    ])
