from setuptools import setup
from setuptools.command.install import install
from setuptools.command.test import test
from setuptools import find_packages
import subprocess, os, sys

install_requires = ["numpy >= 1.5.0", "scipy", "imageio", "scikit-learn","torch","tqdm", "torchvision", "matplotlib"]
setup_requires = ['pytest-runner', 'pytest', 'pytest-cov']

def read_version(fname):
    import re
    VERSIONFILE=fname
    verstrline = open(VERSIONFILE, "rt").read()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))
    return verstr

__version__ = read_version("meanshift/__version__.py")

setup(
      name = 'meanshift',
      author = 'Copyright (C) 2022',
      author_email = 'tbd',
      version = __version__,
      install_requires=install_requires,
      tests_require=setup_requires,
      setup_requires=setup_requires,
      python_requires='>3.6',
      license = 'BSD 3-Clause',
      platforms=["OS Independent"],
      keywords=['meanshift'],
      classifiers=[
          'Development Status :: Beta',
          'License :: BSD 3-Clause',
          'Operating System :: Linux',
          'Programming Language :: Python',
      ],
      packages=find_packages(),
      )
