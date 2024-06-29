from setuptools import setup, find_packages
import supertts

setup(
    name='super-monotonic-align',
    version=supertts.__version__,
    packages=find_packages(include=['super_monotonic_align', 'super_monotonic_align.*'])
)
