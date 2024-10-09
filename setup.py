from setuptools import setup, find_packages


setup_args = dict(
	packages		= find_packages(where = '.'),
	scripts			= ['test.sh']
)

setup(**setup_args)
