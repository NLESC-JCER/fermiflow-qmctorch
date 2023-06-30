## General information

This code is a fork of [FermiFlow](https://github.com/buwantaiji/FermiFlow), extended to 3D molecules with the use of [QMCTorch](https://github.com/NLESC-JCER/QMCTorch). Note that QMCTorch also needs to be installed to be able to use its orbitals and nuclear potential.
The code requires python >= 3.6, <=3.8 and PyTorch 1.8.2 LTS. A GPU support is highly recommended. The transformation of fermion coordinates is implemented as a continuous normalizing flow, making use of the differentiable ODE solver [torchdiffeq](https://github.com/rtqichen/torchdiffeq) with O(1) memory consumption.

Run `python src/Fermion_3D_base.py --help` to check out the available parameters and options for the zero-temperature variational Monte Carlo (VMC) code of a 3D molecule. Below is a simple example:

```python
python src/Fermion_3D_base.py 'He 0 0 0' --nup 1 --ndown 1 --batch 2000 --iternum 1000 --cuda 0
```

The backflow transformation consists of two potentials, which are each represented by a MLP with one hidden layer of 50 nodes.
In the file src/Fermion_3D_base.py (lines 88 and 93), the MLP architectures can be adjusted. As of now this cannot be done through the options.

## Installation

Clone the repository and install the code from source or use the Python package manager:

    git clone https://github.com/NLESC-JCER/fermiflow-qmctorch/
    cd fermiflow-qmctorch
    pip install -e ./
