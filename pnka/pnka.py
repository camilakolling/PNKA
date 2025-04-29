from typing import overload
import torch
import numpy as np


@overload
def pnka(Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    ...

@overload
def pnka(Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    ...

def pnka(Y, Z):
    """Pointwise Normalized Kernel Alignment (PNKA).

    Parameters
    ----------
    Y : torch.Tensor or np.ndarray
        A representation of n input points.
    Z : torch.Tensor or np.ndarray
        A different representation of the same n points.

    Returns
    -------
    torch.Tensor or np.ndarray
        A vector with per-point similarity scores for each point in Y and Z.
    """
    Y = zero_center(Y)
    Z = zero_center(Z)

    K_Y = linear_kernel(Y, Y)
    K_Z = linear_kernel(Z, Z)

    similarity = K_Y @ K_Z.T
    if isinstance(K_Y, torch.Tensor):
        norm_func = torch.linalg.norm
    else:
        norm_func = np.linalg.norm
    norm_K_Y = norm_func(K_Y, axis=1)
    norm_K_Z = norm_func(K_Z, axis=1)
    norm = (norm_K_Y * norm_K_Z)
    return (similarity / norm).diagonal()


@overload
def zero_center(X: torch.Tensor) -> torch.Tensor:
    ...

@overload
def zero_center(X: np.ndarray) -> np.ndarray:
    ...

def zero_center(X):
    return X - X.mean(axis=0)

@overload
def linear_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    ...

@overload
def linear_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    ...

def linear_kernel(X, Y):
    n = X.shape[0]
    return X.reshape(n, -1) @ Y.reshape(n, -1).T
