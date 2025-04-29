import numpy as np
import torch

from pnka import (
    pnka,
    zero_center,
)


def test_zero_center_np():
    assert (
        zero_center(np.array([
            [1, 1],
            [2, 2],
            [3, 3],
        ]))
        == np.array([
            [-1, -1],
            [0, 0],
            [1, 1],
        ])
    ).all()

def test_zero_center_torch():
    assert (
        zero_center(torch.Tensor([
            [-2, -2],
            [2, 2],
            [6, 6],
        ]))
        == torch.Tensor([
            [-4, -4],
            [0, 0],
            [4, 4],
        ])
    ).all()

def test_pnka_scale_invariance_torch():
    torch.random.manual_seed(4329)
    Y = torch.randn(50, 10)
    Z = 5.5 * Y
    torch.testing.assert_close(pnka(Y, Z), torch.ones(50))

def test_pnka_shift_invariance_np():
    rng = np.random.default_rng(532)
    Y = rng.random((50, 10))
    Z = Y + 1
    np.testing.assert_allclose(pnka(Y, Z), np.ones(50))

def test_pnka_symmetry_np():
    rng = np.random.default_rng(16)
    Y = rng.random((50, 10))
    Z = rng.random((50, 15))
    np.testing.assert_allclose(pnka(Y, Z), pnka(Z, Y))

def test_pnka_higher_dim_torch():
    torch.random.manual_seed(6903)
    Y = torch.randn(1000, 10, 8, 8)
    Z = torch.randn(1000, 16, 6, 8)
    result = pnka(Y, Z)
    # Just test that it runs without error
    assert True
