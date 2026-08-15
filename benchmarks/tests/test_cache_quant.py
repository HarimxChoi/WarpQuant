import unittest

import numpy as np
import torch

from tqi3.cache_quant import (
    CacheQuantConfig,
    TurboQuantAttention,
    _mse_quantize,
    _rotation,
    packed_cache_bytes_per_token,
    quantize_values,
    resident_cache_bytes,
    sphere_lloyd_levels,
)


class Module:
    num_key_value_groups = 2
    layer_idx = 0
    training = False


class CacheQuantTest(unittest.TestCase):
    def test_sphere_codebooks_are_ordered_and_symmetric(self):
        for dim in (128, 256):
            for bits in (2, 3, 4):
                levels = sphere_lloyd_levels(dim, bits)
                self.assertTrue(np.all(np.diff(levels) > 0))
                np.testing.assert_allclose(levels, -levels[::-1], atol=1e-7)

    def test_mse_improves_with_bits(self):
        generator = torch.Generator().manual_seed(7)
        values = torch.randn((64, 128), generator=generator)
        rotation = _rotation(128, 11, torch.device("cpu"))
        error2 = torch.mean((_mse_quantize(values, 2, rotation) - values) ** 2)
        error4 = torch.mean((_mse_quantize(values, 4, rotation) - values) ** 2)
        self.assertLess(float(error4), float(error2))

    def test_value_error_improves_with_bits(self):
        generator = torch.Generator().manual_seed(9)
        values = torch.randn((2, 4, 16, 128), generator=generator)
        error2 = torch.mean((quantize_values(values, 2, 32) - values) ** 2)
        error4 = torch.mean((quantize_values(values, 4, 32) - values) ** 2)
        self.assertLess(float(error4), float(error2))

    def test_full_residual_matches_eager_attention(self):
        generator = torch.Generator().manual_seed(13)
        query = torch.randn((1, 4, 8, 128), generator=generator)
        key = torch.randn((1, 2, 8, 128), generator=generator)
        value = torch.randn((1, 2, 8, 128), generator=generator)
        mask = torch.full((1, 1, 8, 8), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        quantizer = TurboQuantAttention(CacheQuantConfig(residual_length=8))
        output, _ = quantizer(Module(), query, key, value, mask, 128**-0.5)
        repeated_key = key[:, :, None].expand(1, 2, 2, 8, 128).reshape(1, 4, 8, 128)
        repeated_value = value[:, :, None].expand(1, 2, 2, 8, 128).reshape(1, 4, 8, 128)
        weights = torch.softmax(
            torch.matmul(query, repeated_key.transpose(2, 3)) * (128**-0.5) + mask,
            dim=-1,
        )
        expected = torch.matmul(weights, repeated_value).transpose(1, 2).contiguous()
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_storage_accounting_includes_metadata(self):
        self.assertEqual(packed_cache_bytes_per_token(128, 3, 4, 32), 136)
        self.assertEqual(packed_cache_bytes_per_token(256, 3, 4, 32), 264)

    def test_resident_storage_keeps_recent_tokens_in_bf16(self):
        self.assertEqual(resident_cache_bytes(512, 128, 32_768, 9_472), 7_831_552)
        self.assertEqual(resident_cache_bytes(64, 128, 32_768, 9_472), 2_097_152)
        self.assertEqual(resident_cache_bytes(512, 0, 32_768, 9_472), 4_849_664)


if __name__ == "__main__":
    unittest.main()


