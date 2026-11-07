import pytest

torch = pytest.importorskip("torch")

from kangaroo.loss import linear_cross_entropy_sum  # noqa: E402


def _case(n=37, h=16, v=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n, h, generator=g, requires_grad=True)
    weight = torch.randn(v, h, generator=g)
    labels = torch.randint(0, v, (n,), generator=g)
    return hidden, weight, labels


@pytest.mark.parametrize("softcap", [None, 30.0])
@pytest.mark.parametrize("chunk", [1, 8, 1000])
def test_chunked_matches_dense_value_and_grad(chunk, softcap):
    hidden, weight, labels = _case()
    dense = linear_cross_entropy_sum(hidden, weight, labels, chunk=0, softcap=softcap)
    (g_dense,) = torch.autograd.grad(dense, hidden)
    chunked = linear_cross_entropy_sum(hidden, weight, labels, chunk=chunk, softcap=softcap)
    (g_chunked,) = torch.autograd.grad(chunked, hidden)
    assert torch.allclose(dense, chunked, rtol=1e-5, atol=1e-5)
    assert torch.allclose(g_dense, g_chunked, rtol=1e-4, atol=1e-5)


def test_matches_torch_reference():
    hidden, weight, labels = _case()
    ref = torch.nn.functional.cross_entropy(hidden @ weight.T, labels, reduction="sum")
    got = linear_cross_entropy_sum(hidden, weight, labels, chunk=8)
    assert torch.allclose(ref, got, rtol=1e-5)


def test_weight_gradient_flows_when_trainable():
    hidden, weight, labels = _case()
    weight = weight.clone().requires_grad_()
    dense = linear_cross_entropy_sum(hidden, weight, labels, chunk=0)
    (gw_dense,) = torch.autograd.grad(dense, weight)
    chunked = linear_cross_entropy_sum(hidden, weight, labels, chunk=5)
    (gw_chunked,) = torch.autograd.grad(chunked, weight)
    assert torch.allclose(gw_dense, gw_chunked, rtol=1e-4, atol=1e-5)


def test_empty_input_is_zero():
    out = linear_cross_entropy_sum(torch.zeros(0, 4, requires_grad=True), torch.randn(5, 4), torch.zeros(0, dtype=torch.long))
    assert out.item() == 0.0


def test_no_grad_path():
    hidden, weight, labels = _case()
    with torch.no_grad():
        out = linear_cross_entropy_sum(hidden, weight, labels, chunk=8)
    assert not out.requires_grad
