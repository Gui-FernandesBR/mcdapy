import setuptools  # import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="mcdapy",
    version="0.1.0",
    packages=[],
    url="https://github.com/Gui-FernandesBR/mcdapy",
    license="GNU GENERAL PUBLIC LICENSE",
    author="Guilherme Fernandes Alves",
    author_email="gf10.alves@gmail.com",
    maintainer="Guilherme Fernandes Alves",
    description="A simple application for Multiple-criteria decision analysis (MCDA) in python.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPL License",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=required,
    install_lib="mcdapy",
)
