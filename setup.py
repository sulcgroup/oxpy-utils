from setuptools import setup, find_packages

setup(
    name='oxpy_utils',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={'oxpy_utils': ['defaults/inputs/*.json']},
    description='Utilities to use the oxDNA (via oxpy) in enhanced sampling and linked groups of simulations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sulcgroup/oxpy-utils',
    author='Joshua Evans',
    author_email='joshuaevanslowell@gmail.com',
    license='MIT',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'pyarrow',
        'networkx',
        'tqdm',
        'PyYAML',
        'safe_exit',
        'numba',
        'pygltflib',
        'biopython',
        'ipywidgets',
        'nvidia-ml-py3',
        'py',
        # oxpy: install separately from oxDNA source (https://lorenzo-rovigatti.github.io/oxDNA/install.html#python-bindings)
        # nupack: install separately from https://nupack.org
    ],
)
