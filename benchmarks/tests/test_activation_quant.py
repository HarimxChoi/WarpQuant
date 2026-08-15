import unittest

import torch

from tqi3.activation_quant import LinearInputQuantizer, dynamic_symmetric_quantize


class ActivationQuantTest(unittest.TestCase):
    def test_zero_input_stays_zero(self):
        values = torch.zeros((2, 8, 128), dtype=torch.bfloat16)
        torch.testing.assert_close(dynamic_symmetric_quantize(values, 4), values)

    def test_int8_error_is_lower_than_int4(self):
        values = torch.randn((8, 32, 128), generator=torch.Generator().manual_seed(17))
        error4 = torch.mean((dynamic_symmetric_quantize(values, 4) - values) ** 2)
        error8 = torch.mean((dynamic_symmetric_quantize(values, 8) - values) ** 2)
        self.assertLess(float(error8), float(error4))

    def test_only_decoder_linear_modules_are_hooked(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
                self.lm_head = torch.nn.Linear(8, 8)

        model = Block()
        quantizer = LinearInputQuantizer(8)
        stats = quantizer.install(model)
        self.assertEqual(stats.modules, 1)
        quantizer.remove()


if __name__ == "__main__":
    unittest.main()


