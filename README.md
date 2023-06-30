The code requires python >= 3.6, <=3.8 and PyTorch 1.8.2 LTS. A GPU support is highly recommended. The transformation of fermion coordinates is implemented as a continuous normalizing flow, where we have used the differentiable ODE solver [torchdiffeq](https://github.com/rtqichen/torchdiffeq) with O(1) memory consumption.

Note that QMCTorch also needs to be installed to be able to use its orbitals and nuclear potential for molecules (https://github.com/NLESC-JCER/QMCTorch).

Run `python src/Fermion_3D_base.py --help` to check out the available parameters and options for the finite-temperature variational Monte Carlo (VMC) code of a 2D quantum dot system. Below is a simple example:

```python
python src/BetaFermionHO2D.py --beta 10.0 --nup 3 --Z 2.0 --deltaE 2.0 --cuda 0 --boltzmann --iternum 1000
```

The corresponding ground-state VMC code [FermionHO2D.py](src/FermionHO2D.py) is very similar.

## To cite

```bibtex
@Article{JML-1-38,
    author = {Hao and Xie and and 22776 and and Hao Xie and Linfeng and Zhang and and 22777 and and Linfeng Zhang and Lei and Wang and and 22778 and and Lei Wang},
    title = {Ab-Initio Study of Interacting Fermions at Finite Temperature with Neural Canonical Transformation},
    journal = {Journal of Machine Learning},
    year = {2022},
    volume = {1},
    number = {1},
    pages = {38--59},
    abstract = {We present a variational density matrix approach to the thermal properties of interacting fermions
in the continuum. The variational density matrix is parametrized by a permutation equivariant many-body
unitary transformation together with a discrete probabilistic model. The unitary transformation is implemented as a quantum counterpart of neural canonical transformation, which incorporates correlation effects
via a flow of fermion coordinates. As the first application, we study electrons in a two-dimensional quantum dot with an interaction-induced crossover from Fermi liquid to Wigner molecule. The present approach
provides accurate results in the low-temperature regime, where conventional quantum Monte Carlo methods
face severe difficulties due to the fermion sign problem. The approach is general and flexible for further extensions, thus holds the promise to deliver new physical results on strongly correlated fermions in the context
of ultracold quantum gases, condensed matter, and warm dense matter physics.},
    issn = {2790-2048},
    doi = {https://doi.org/10.4208/jml.220113},
    url = {http://global-sci.org/intro/article_detail/jml/20371.html}
}
