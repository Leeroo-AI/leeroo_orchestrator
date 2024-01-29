from setuptools import setup

setup( 
    name="leeroo_orchestrator",
    install_requires=[
        'boto3==1.34.29',
        'openai==0.27.7',
        'paramiko==3.4.0',
        'python-dotenv==1.0.1',
        'requests==2.31.0',
        'sagemaker==2.205.0',
    ],
    py_modules=[],
)