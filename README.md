# Pointwise Normalized Kernel Alignment (PNKA)

In [Investigating the Effects of Fairness Interventions Using Pointwise Representational Similarity](https://openreview.net/pdf?id=CkVlt2Qgdb) (TMLR, 2025) we propose Pointwise Normalized Kernel Alignment (PNKA) as an instance of a measure that quantifies how similarly an *individual* input is represented in two representation spaces.

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
@article{
  kolling2025investigating,
  title={Investigating the Effects of Fairness Interventions Using Pointwise Representational Similarity},
  author={Camila Kolling and Till Speicher and Vedant Nanda and Mariya Toneva and Krishna P. Gummadi},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=CkVlt2Qgdb},
  note={}
}
```