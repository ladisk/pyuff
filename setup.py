import os
import re
from setuptools import setup

regexp = re.compile(r'.*__version__ = [\'\"](.*?)[\'\"]', re.S)

base_path = os.path.dirname(__file__)

init_file = os.path.join(base_path, 'pyuff', '__init__.py')
with open(init_file, 'r') as f:
    module_content = f.read()

    match = regexp.match(module_content)
    if match:
        version = match.group(1)
    else:
        raise RuntimeError(
            'Cannot find __version__ in {}'.format(init_file))


with open(os.path.join(base_path, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def parse_requirements(filename):
    ''' Load requirements from a pip requirements file '''
    with open(filename, 'r') as fd:
        lines = []
        for line in fd:
            line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines

requirements = parse_requirements('requirements.txt')

setup(name='pyuff',
      version=version,
      author='Primož Čermelj, Janko Slavič',
      author_email='primoz.cermelj@gmail.com, janko.slavic@fs.uni-lj.si',
      description='UFF (Universal File Format) read/write.',
      url='https://github.com/ladisk/pyuff',
      packages=['pyuff', 'pyuff.datasets'],
      long_description=long_description,
      install_requires=requirements,
      keywords='UFF, UNV, Universal File Format, read/write',
      )
