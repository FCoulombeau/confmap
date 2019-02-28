#!/usr/bin/env python

# This will try to import setuptools. If not here, it will reach for the embedded
# ez_setup (or the ez_setup package). If none, it fails with a message
import sys
from codecs import open

try:
    from setuptools import find_packages, setup
    from setuptools.command.test import test as TestCommand
except ImportError:
    try:
        import ez_setup
        ez_setup.use_setuptools()
    except ImportError:
        raise ImportError('ConfMap could not be installed, probably because'
            ' neither setuptools nor ez_setup are installed on this computer.'
            '\nInstall ez_setup ([sudo] pip install ez_setup) and try again.')


class PyTest(TestCommand):
    """Handle test execution from setup."""

    user_options = [('pytest-args=', 'a', "Arguments to pass into pytest")]

    def initialize_options(self):
        """Initialize the PyTest options."""
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def finalize_options(self):
        """Finalize the PyTest options."""
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        """Run the PyTest testing suite."""
        try:
            import pytest
        except ImportError:
            raise ImportError('Running tests requires additional dependencies.'
                '\nPlease run (pip install moviepy[test])')

        errno = pytest.main(self.pytest_args.split(" "))
        sys.exit(errno)


cmdclass = {'test': PyTest} # Define custom commands.

if 'build_docs' in sys.argv:
    try:
        from sphinx.setup_command import BuildDoc
    except ImportError:
        raise ImportError('Running the documenation builds has additional'
            ' dependencies. Please run (pip install moviepy[docs])')

    cmdclass['build_docs'] = BuildDoc

__version__ = None # Explicitly set version to quieten static code checkers.
exec(open('confmap/version.py').read()) # loads __version__

# Define the requirements for specific execution needs.
requires = [
    "matplotlib>=2.0.0,<3.0; python_version>='3.4'",
    'numpy',
    'moviepy'
    ]


doc_reqs = [
        'numpydoc>=0.6.0,<1.0',
        'sphinx_rtd_theme>=0.1.10b0,<1.0', 
        'Sphinx>=1.5.2,<2.0',
    ]

test_reqs = [
#        'coveralls>=1.1,<2.0',
#        'pytest-cov>=2.5.1,<3.0',
        'pytest>=3.0.0,<4.4'
    ]

extra_reqs = {
    "optional": [],
    "doc": doc_reqs,
    "test": test_reqs
    }

# Load the README.
with open('README.rst', 'r', 'utf-8') as f:
    readme = f.read()

setup(
    name='confmap',
    version=__version__,
    author='F.Coulombeau 2019',
    author_email="coulombeau@gmail.com",
    description='Conformal mapping on images and videos',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://fcoulombeau.frama.io/git/transconf/',
    license='MIT License',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English, French',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Multimedia',
        'Topic :: Multimedia :: Video',
        'Topic :: Multimedia :: Image',
    ],
    keywords='conformal mapping complex analysis',
    packages=find_packages(exclude='docs'),
    cmdclass=cmdclass,
    command_options={
        'build_docs': {
            'build_dir': ('setup.py', './docs/build'),
            'config_dir': ('setup.py', './docs'),
            'version': ('setup.py', __version__.rsplit('.', 2)[0]),
            'release': ('setup.py', __version__)}},
    tests_require=test_reqs,
    install_requires=requires,
    extras_require=extra_reqs,
)