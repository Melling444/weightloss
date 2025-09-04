import setuptools

from weightloss import __version__

setuptools.setup(
    name='weightloss',
    version=__version__,
    description='Weight Loss Tracking and Data Analysis',
    author='John Melling',
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'seaborn',
        'plotly',
        'scikit-learn',
        'duckdb',
        'boto3',
        'shiny==1.4.0',
    ],
    packages=setuptools.find_packages(include=['weightloss', 'weightloss.*']),
    zip_safe=False
)