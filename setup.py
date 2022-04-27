import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Ultrasonic random forests",
    version = "0.0.1",
    author = "Frank Alexander Ihle",
    author_email = "andrewjcarter@gmail.com",
    description = ("A reimplementation of the algorithm that took part of the ETH CLUST challenge"),
    license = "MIT",
    keywords = "machine learning random forest ultrasound tracking CLUST MICCAI",
    url = "tbd",
    install_requires=[
        'glob2==0.7',
        'sklearn==0.0',
        'numpy==1.22.3',
        'opencv-python==4.5.5.64'],
    packages=find_packages(),
    long_description=read('README.md'),
    entry_points={
        'console_scripts': [
            'USRF=USRF.main:main'
        ]
    }
)