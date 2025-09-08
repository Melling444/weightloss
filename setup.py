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
        'google-ai-generativelanguage==0.6.15',
        'google-api-core==2.24.2',
        'google-api-python-client==2.170.0',
        'google-auth==2.40.2',
        'google-auth-httplib2==0.2.0',
        'google-generativeai==0.8.5',
        'googleapis-common-protos==1.70.0',
        'python-dotenv==1.1.0',
    ],
    packages=setuptools.find_packages(include=['weightloss', 'weightloss.*']),
    zip_safe=False
)