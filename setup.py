from setuptools import setup, find_packages

setup(
    name='ml_core',
    version='0.1',
    description='Core machine learning algorithms implemented in Python',
    author='Your Name',
    author_email='shorif.mahmud.abdullah@gmail.com',
    url='https://github.com/shorif10/ml_core',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
