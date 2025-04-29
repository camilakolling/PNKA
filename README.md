# Pointwise Representation Similarity

In [Pointwise Representation Similarity](https://arxiv.org/abs/2305.19294) we propose Pointwise Normalized Kernel Alignment (PNKA) as an instance of a measure that quantifies how similarly an *individual* input is represented in two representation spaces.

## Installation

Create and activate a [virtual environment](https://docs.python.org/3/library/venv.html), then run the following in the root directory to install dependencies:
```bash
pip install .
```

Alternatively, you can also use [Poetry](https://python-poetry.org/) to install dependencies by running
```bash
poetry install
```
in the root directory.


## Usage

```python
import torch

from pnka import pnka

# Replace Y and Z with two sets of representations
torch.manual_seed(1337)
Y = torch.randn(100, 10)
Z = torch.randn(100, 10) + Y

# Compute a vector of pointwise similarity scores
pointwise_similarities = pnka(Y, Z)
# The similarity score of the first point
first_similarity_score = pointwise_similarities[0]
# An aggregate of the representation similarity across all points
aggregated_similarity = pointwise_similarities.mean()
```

## Citation

If you find our work useful, please cite it:

```
@article{kolling2023pointwise,
  title={Pointwise Representational Similarity},
  author={Kolling, Camila and Speicher, Till and Nanda, Vedant and Toneva, Mariya and Gummadi, Krishna P},
  journal={arXiv preprint arXiv:2305.19294},
  year={2023}
}
```