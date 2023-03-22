from setuptools import find_packages
from setuptools import setup,Extension

setup(name="CASTEPbands",
      version="0.0.2",
      packages=find_packages(),
      description="CASTEP module for plotting band structures and phonon dispersions.",
      url="https://github.com/zachary-hawk/CASTEP_bands.git",
      author="Zachary Hawkhead",
      author_email="zachary.hawkhead@ymail.com",
      license="MIT",
      install_requires=["numpy",
                        "matplotlib",
                        "ase>=3.18.1"],
      )



