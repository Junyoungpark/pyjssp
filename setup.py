from setuptools import setup, find_packages

setup(
    name='jsspsimulator',
    version='0.1',
    description='A native python JSSP simulator',
    author='Junyoung Park, Jaehyeong Chun',
    author_email='Junyoungpark@kaist.ac.kr',
    url='https://github.com/Junyoungpark/JSSPsimulator',
    install_requires=['numpy', 'matplotlib', 'plotly', 'networkx'],
    packages=find_packages(),
    keywords=['JSSP', 'JSSP simulator'],
    python_rquires='>=3',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ]
)
