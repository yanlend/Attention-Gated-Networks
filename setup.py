#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(name='AttentionGatedNetworks',
      version='1.0',
      description='Pytorch library for Soft Attention',
      long_description=readme,
      author='Ozan Oktay & Jo Schlemper',
      install_requires=[
        "numpy",
        "torch",
        "opencv-python",
        "matplotlib",
        "scipy",
        "torchvision",
        "tqdm",
        "visdom",
        "nibabel",
        "scikit-image",
        "scikit-learn",
        "h5py",
        "pandas",
        "dominate",
        'torchsample @ https://github.com/ozan-oktay/torchsample/tarball/master#egg=torchsample-0.1.3',
      ],
      packages=find_packages(exclude=('tests', 'docs'))
)

