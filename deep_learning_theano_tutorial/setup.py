try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

config = {
	'description': 'CNN',
	'author': 'Viviana Petrescu',
	'author_email': 'My email',
	'name': 'CNN'
}

setup(**config)
