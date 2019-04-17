import setuptools


setuptools.setup(
    author='Ounkomol, Chek and Fernandes, Daniel A. and Seshamani, Sharmishtaa and Maleckar, Mary M. and Collman, Forrest and Johnson, Gregory R.',
    author_email='gregj@alleninstitute.org',
    description='A machine learning model for transforming microsocpy images between modalities',
    entry_points={
        'console_scripts': [
            'fnet = fnet.cli.main:main',
        ]
    },
    name='fnet',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    url='https://github.com/AllenCellModeling/pytorch_fnet',
    version="1.0",
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'tifffile',
        'torch>=1.0',
        'tqdm',
        'scikit-image>=0.15.0',
    ],
    extras_require={
        'dev': [
            'flake8',
            'pylint',
            'pytest',
            'pytest-cov',
        ],
        'examples': [
            't4==0.0.10',
        ]
    },
)
