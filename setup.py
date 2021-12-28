import os
from setuptools import setup

setup(
    name="Web Spam Detector",
    version="1.0.0",
    description="Classification of web pages spamicity using logistic regression",
    long_description=open(os.path.abspath(os.path.dirname(__file__)),"README.md"),
    long_description_content_type="text/markdown",
    author="YOUSSEF AIDANI",
    install_requires=["setuptools"]
)