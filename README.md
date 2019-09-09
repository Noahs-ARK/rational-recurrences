# Rational Recurrences

PyTorch implementation for [Rational Recurrences](https://homes.cs.washington.edu/~hapeng/paper/peng2018rational.pdf).

#### Reference:
```
@InProceedings{peng2018rational,
  author = {Peng, Hao and Schwartz, Roy and Thomson, Sam and Smith, Noah A.},
  title     = {Rational Recurrences},
  booktitle = {In Proceedings of the Conference on Empirical Methods in Natural Language Processing},
  year      = {2018}
}
```
<br>
## Setup

Install Anaconda 4.6.2.

To install this project's dependencies into a conda virtualenv:
```bash
conda env create -f environment.yml
conda activate rational-recurrences
conda install pytorch=0.3.1 cuda80 -c pytorch
pip install pynvrtc
```

