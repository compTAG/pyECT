from setuptools import setup, find_packages

setup(
    name='pyect',
    version='0.1.3',
    author='Alex McCleary, Eli Quist, Jack Ruder, Jacob Sriraman',
    author_email='eli.quist@student.montana.edu',
    description='Generalized computation of the WECT using PyTorch.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/compTAG/pyECT',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch',
        'Pillow',
        'numpy',
        'torchvision'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
