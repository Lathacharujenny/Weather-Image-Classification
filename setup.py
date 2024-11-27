from setuptools import setup, find_packages

HYPEN_E_DOT = '-e .'

def get_requirements(filepath):
    requirements = []
    with open(filepath, 'r') as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements]
        if HYPEN_E_DOT:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='Weather-Image-Classifaction',
    version='0.0.1',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt'),
    author='Latha',
    author_email='lathacharugundla@gmail.com'
)