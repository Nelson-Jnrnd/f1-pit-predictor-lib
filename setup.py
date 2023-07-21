from setuptools import find_packages, setup

setup(
    name='f1pitpred',
    packages=find_packages(),
    version='0.1.28',
    description='Predicting pitstops in F1',
    author='Nelson Jeanrenaud',
    license='MIT',
    install_requires=['numpy', 'pandas','matplotlib','scikit-learn'],
)