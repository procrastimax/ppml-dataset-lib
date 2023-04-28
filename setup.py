from setuptools import find_packages, setup


setup(
    name='ppml_datasets',
    packages=find_packages(include=['ppml_datasets']),
    python_requires='>=3.8.6',
    version='0.1.0',
    description='a library that contains dataset structures and functions for privacy preserving machine learning',
    author='procrastimax',
    license='MIT',
    install_requires=[
        "tensorflow==2.12.0",
        "tensorflow-datasets==4.9.2",
        "gdown==4.7.1",
        "scikit-learn==1.2.2",
        "matplotlib==3.7.1"],
    setup_requires=[''],
)
