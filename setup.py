from setuptools import setup, find_packages

print("Packages: {}".format(find_packages()))
setup(
    name = "dirigo",
    description="Laser scanning imaging control software",
    version = "0.1",
    packages = find_packages(),
    author = "Timothy D. Weber",
    author_email = "tweber@mit.edu",
)