# Author
[Dario Mameli] (dario.mameli@ugent.be)

# Project structure
1) **dataset**: contains the dataset files.
2) **src**: contains the source code.

The code was developed using Python 3.9.
The code has been tested on the following machine configuration:
1) Windows 10
2) CPU 11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz   3.11 GHz 
3) RAM 16GB
4) GPU is present but not used


# Code structure
1) **main.py**:
main function that performs the analysis given the methods in _method.py_.
2) **ContentBasedRecommender.py**:
function containing the definition of the methods to call for the analysis.

# How to run
The file to run is _main.py_

**NOTE**: It may be needed to adjust the relative paths to the datasets (set in the beginning of _main.py_ source file) if those files are not present in the dataset directory.