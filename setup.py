from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '1.0.0'

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='akdprframework',
    version=__version__,
    description='AKDPRFramework is a framework for doing fundamental deep learning research.',
    url='https://github.com/theroyakash/AKDPRFramework',
    download_url='https://github.com/theroyakash/AKDPRFramework/tarball/main',
    license='Apache License',
    packages=find_packages(),
    include_package_data=True,
    author='theroyakash',
    install_requires=install_requires,
    setup_requires=['numpy', 'scipy'],
    dependency_links=dependency_links,
    author_email='royakashappleid@icloud.com'
)