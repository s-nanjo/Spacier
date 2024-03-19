from setuptools import setup, find_packages

setup(
    name='spacier',
    version='0.3.0',
    author='Shun Nanjo, Arifin',
    author_email='nanjos@ism.ac.jp, arifin@ism.ac.jp',
    description='A library for efficient exploration of materials space,'
                'focusing on polymers.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/s-nanjo/Spacier',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
    ],
    extras_require={
        'advanced': [
            'GPy',
            'torch',
            'gpytorch',
            'psutil',
            'matplotlib',
            'rdkit>=2020.03',
            'mdtraj>=1.9',
            'lammps>=2020.03.03',
            'radonpy-pypi',
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
