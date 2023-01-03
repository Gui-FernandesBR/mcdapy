# mcdapy

A simple application for **Multiple-criteria decision analysis (MCDA)** in python.

Powered by the [Flet](https://flet.dev/) package, which allows creating flutter applications totally on python.

Two methods are currently implemented: PATTERN and [ELECTRE](https://doi.org/10.1016/j.ejor.2015.07.019).

The usage of the application is very simple, just enter the data and the method you want to use, and the application will return the results.

![demo](https://raw.githubusercontent.com/Gui-FernandesBR/mcdapy/master/docs/source/static/g1.gif)

## Installation

In the terminal, run the following command:

```shell
pip install git+https://github.com/Gui-FernandesBR/mcdapy.git
```

The package will be installed from the master branch of mcdapy project directly
in your python environment.
It is recommended to use a virtual environment.

## How to use

As a python script:

```python
import mcdapy

app = mcdapy.App()
```

On your terminal:

```shell
python cli.py
```

As an executable:

```shell
python pyinstaller --onefile cli.py --name mcdapy
```

## Directory structure

The github repository is organized as follows:

```shell
mcdapy
├── mcdapy
│   ├── methods
│   │   ├── __init__.py
│   │   ├── pattern.py
│   │   ├── electre.py
│   ├── __init__.py
│   ├── app.py
├── examples
│   ├── methods.ipynb
│   ├── e1.gif
├── cli.py
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-dev.txt
├── setup.py
```

## Contributing

Pull requests are welcome at mcdapy. For major changes, please open an issue first to discuss what you would like to change.

You can also raise an issue if you find any bug or have any suggestion.
