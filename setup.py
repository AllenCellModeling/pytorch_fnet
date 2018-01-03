#!/usr/bin/env python
from setuptools import setup
import sys

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(name='pytorch_fnet',
      description=' a machine learning model for transforming microsocpy images between modalities',
      author='Chek Ounkomol, Greg Johnsons',
      author_email='gregj@alleninstitute.org',
      url='https://github.com/AllenCellModeling/pytorch_fnet',
      packages=['fnet'],
      install_requires=required)
