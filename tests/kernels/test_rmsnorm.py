import pytest
import torch

from conftest import BF16, DEVICE, DTYPE, close


def reference(x, w, eps):
    wide = x.float()
    return wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + eps) * w.float()


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("y_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows,dim", [(1, 2048), (7, 2048), (33, 4096), (5, 100)])
def test_rmsnorm(lx, x_dtype, y_dtype, rows, dim):
    torch.manual_seed(0)
    x = torch.randn(rows, dim, device=DEVICE).to(x_dtype)
    w = (torch.rand(dim, device=DEVICE) + 0.5).to(torch.bfloat16)
    y = torch.empty(rows, dim, device=DEVICE, dtype=y_dtype)
    lx("rmsnorm", x, DTYPE[x_dtype], w, y, DTYPE[y_dtype], rows, dim, 1e-6)
    want = reference(x, w, 1e-6)
    close(y, want.to(y_dtype), atol=1e-5 if y_dtype == torch.float32 else 1e-2, rtol=1e-2 if y_dtype == torch.bfloat16 else 1e-5)
