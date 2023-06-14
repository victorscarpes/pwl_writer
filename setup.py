from setuptools import find_packages, setup

setup(name="pwl_writer",
      version="1.0.0",
      description="Package to easily write pwl files in terms of durations and not instants.",
      author="Victor Sabiá Pereira Carpes",
      url="https://github.com/victorscarpes/pwl_writer",
      packages=find_packages(),
      install_requires=["numpy"]
      )
