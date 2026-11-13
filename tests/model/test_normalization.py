"""Match the public RMSNorm precision policy, including learned BF16 scales."""

from contextlib import nullcontext

import pytest

torch = pytest.importorskip("torch")

from moeloop.model import RMSNorm


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("amp", [False, True])
def test_rms_norm_matches_public_output_dtype_and_gradients(device, weight_dtype, amp):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for the native autocast precision policy")
    dim = 672
    actual = RMSNorm(dim, eps=1e-6).to(device=device, dtype=weight_dtype)
    reference = torch.nn.RMSNorm(dim, eps=1e-6).to(device=device, dtype=weight_dtype)
    with torch.no_grad():
        scales = torch.linspace(.7, 1.3, dim, device=device)
        actual.weight.copy_(scales)
        reference.weight.copy_(scales)
    values = torch.linspace(-3, 3, 2 * 5 * dim, device=device).reshape(2, 5, dim).bfloat16()
    x = values.clone().requires_grad_()
    expected_x = values.clone().requires_grad_()
    context = torch.autocast(device_type=device, dtype=torch.bfloat16) if amp else nullcontext()
    with context:
        output, expected = actual(x), reference(expected_x)
    assert output.dtype == expected.dtype
    # Identical standard forward semantics should not introduce another BF16
    # rounding of the normalization or learned scale multiplication.
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
    upstream = torch.linspace(.2, 1.7, values.numel(), device=device).reshape_as(values)
    (output.float() * upstream).sum().backward()
    (expected.float() * upstream).sum().backward()
    torch.testing.assert_close(x.grad, expected_x.grad)
    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
