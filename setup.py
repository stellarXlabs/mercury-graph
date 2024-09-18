from setuptools import setup, find_packages


setup_args = dict(
	packages		= find_packages(where = 'mercury'),
	package_dir		= {'' : 'mercury'},
	scripts			= ['mercury/test_all.py']
)

setup(**setup_args)
