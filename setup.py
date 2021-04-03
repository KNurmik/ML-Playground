import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ML Playground",
    version="0.0.1",
    author="Karl Hendrik Nurmeots",
    author_email="",
    description="Implementing machine learning and statistical models from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'autograd',
        'pandas',
        'matplotlib',
        'torch',
        'tqdm'
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)