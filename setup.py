
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LGNpy", 
    version="0.0.1",
    author="Prasad Ostwal",
    author_email="prasadostwal@gmail.com",
    description="Linear Gaussian Networks - Inference, Parameter Learning and Representation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ostwalprasad/lgnpy",
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 5 - Production/Stable'
    ],
    keywords=' '.join(['linear', 'gaussian', 'networks', 'representation','inference']),
    install_requires=['numpy', 'pandas', 'matplotlib', 'seaborn','networkx'],
    python_requires='>=3.6',
)

