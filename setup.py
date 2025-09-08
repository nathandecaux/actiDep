from setuptools import setup, find_packages

setup(
    name='actiDep',
    version='0.1.0',
    author='Nathan Decaux',
    author_email='nathan.decaux@irisa.fr',
    description='Package de traitement de la base de données ActiDep',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nathandecaux/ActiDep',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        # Ajoutez d'autres dépendances selon vos besoins
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)