from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='fair21',  # TODO! change to plain old fair after demo
    version='2.1.0',
    description='Finite-amplitude Impulse Response (FaIR) simple climate model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/OMS-NetZero/FAIR',
    author='Chris Smith',  # TODO: add in all contribs
    author_email='c.j.smith1@leeds.ac.uk',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='simple, climate, model, temperature, CO2, forcing, emissions',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.6, <4',
    # this environment is installed with conda, but need to pip install to develop
    # install_requires=[
        # 'climateforcing',
        # 'jupyter',
        # 'matplotlib',
        # 'nbstripout',
        # 'netcdf-scm',
        # 'numpy',
        # 'pandas',
        # 'scipy',
        # 'scmdata',
        # 'seaborn',
        # 'tqdm',
    # ],
)