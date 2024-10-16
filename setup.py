from setuptools import setup, find_packages


setup_args = dict(
	packages		= find_packages(include='mercury*', exclude=['docker', 'unit_tests']),
	scripts			= ['test.sh']
)

setup(**setup_args)
