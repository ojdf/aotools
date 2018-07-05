from distutils.core import setup
import versioneer

# I really prefer Markdown to reStructuredText.  PyPi does not.  This allows me
# to have things how I'd like, but not throw complaints when people are trying
# to install the package and they don't have pypandoc or the README in the
# right place.
try:
   import pypandoc
   long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   long_description = ''

setup(
    name='aotools',
    author_email='a.p.reeves@durham.ac.uk',
    packages=['aotools',
              'aotools.astronomy',
              'aotools.functions',
              'aotools.image_processing',
              'aotools.turbulence',
              'aotools.wfs',
              ],
    description='A set of useful functions for Adaptive Optics in Python',
    long_description=long_description,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass()
)
