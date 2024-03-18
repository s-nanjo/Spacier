from setuptools import setup, find_packages

setup(
    name='spacier',
    version='0.3.0',
    author='Your Name',
    author_email='youremail@example.com',
    description='A library for efficient exploration of materials space,'
                'focusing on polymers.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/spacier',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.5',
        'pandas>=1.0.5',
        'scipy>=1.5.0',
        'scikit-learn>=0.23.1',
    ],
    extras_require={
        'advanced': [
            'GPy>=1.9.9',
            'torch>=1.5.0',
            'gpytorch>=1.2.0'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
