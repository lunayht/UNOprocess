# Non-Contrastive Self-Supervised Learning with UNO Process for Respiratory Sound Analysis
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CodeFactor](https://www.codefactor.io/repository/github/lunayht/unoprocess/badge)](https://www.codefactor.io/repository/github/lunayht/unoprocess)

This repository is the official implementation of Non-Contrastive Self-Supervised Learning with **U**npredictable **N**euron **O**peration (UNO) Process for Respiratory Sound Analysis. 

## Conda Environment
0. Install `conda` environment:
```setup
conda env create -f env.yml
```
1. Activate the environment:
```bash
conda activate uno_pl
```
## Dataset
Please refer to `icbhi_data/` for the data preparation and preprocessing.

## Acknowledgements
1. AST model: https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py
2. ResNet model: https://github.com/hche11/VGGSound/blob/master/models/resnet.py 
2. Non-Constrastive Self-Supervised Learning: https://github.com/lucidrains/byol-pytorch