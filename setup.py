import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='micrograd',
    version='0.0.1',
    author='Naveed',
    author_email='sknaveed513@gmail.com',
    url='https://github.com/Naveed513/micrograd',
    description='A small scalar-based autograd with a tiny PyTorch based neural network library on top',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages()
)