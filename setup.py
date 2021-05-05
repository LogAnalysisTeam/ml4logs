from setuptools import find_packages, setup

setup(
    name='ml4logs',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    version='0.2.0',
    description='Machine Learning methods for log file processing',
    author='Log Analysis Team, AIC, FEE CTU in Prague',
    license='MIT',

    install_requires=[
        'requests',
        'numpy',
        'pandas',
        'fasttext',
        'drain3',
        'sklearn', 
        'pyod',
        'torch',
        'tqdm',
        'tqdm-logging-wrapper'
    ]
)
