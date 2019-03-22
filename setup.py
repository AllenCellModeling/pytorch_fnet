import setuptools


setuptools.setup(
    name='pytorch_fnet',
    version="1.0",
    description='A machine learning model for transforming microsocpy images between modalities',
    author='Ounkomol, Chek and Fernandes, Daniel A. and Seshamani, Sharmishtaa and Maleckar, Mary M. and Collman, Forrest and Johnson, Gregory R.',
    author_email='gregj@alleninstitute.org',
    url='https://github.com/AllenCellModeling/pytorch_fnet',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'pytest',
        'scipy',
        'tifffile',
        'torch==0.4',
        'tqdm',
    ]
)
