from distutils.core import setup
import versioneer

setup(
    name='aotools',
    author_email='a.p.reeves@durham.ac.uk',
    packages=[  'aotools',
                'aotools.circle',
                'aotools.centroiders',
                'aotools.phasescreen',
                'aotools.fft',
                'aotools.interp',
                'aotools.wfs'
                ],

    description='A set of useful functions for Adaptive Optics in Python',
    long_description=open('README.md').read(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass()
)
