from pip.req import parse_requirements
from setuptools import find_packages, setup

install_reqs = parse_requirements('requirements.txt')

reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='dl_hpc_starter_pack',
    entry_points={
      'console_scripts': [
          'dl_hpc_starter_pack = dl_hpc_starter_pack:main',
      ],
    },
    packages=find_packages('src'),
    package_dir={'': 'src'},
    version='0.1.0',
    url='https://github.com/csiro-mlai/dl_hpc_starter_pack',
    description='Deep Learning and HPC Starter Pack',
    author='CSIRO',
    install_requires=reqs
    license='GNU General Public License v3.0',
)
