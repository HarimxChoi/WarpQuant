from dataclasses import dataclass

import torch


def dynamic_symmetric_quantize(values: torch.Tensor, bits: int) -> torch.Tensor:
    if bits not in (4, 8):
        raise ValueError("activation bits must be four or eight")
    source = values.float()
    limit = (1 << (bits - 1)) - 1
    scale = (source.abs().amax(dim=-1, keepdim=True) / limit).clamp_min(1e-12)
    codes = torch.round(source / scale).clamp(-limit, limit)
    return (codes * scale).to(values.dtype)


@dataclass
class ActivationQuantStats:
    modules: int = 0
    calls: int = 0
    values: int = 0


class LinearInputQuantizer:
    def __init__(self, bits: int):
        self.bits = bits
        self.stats = ActivationQuantStats()
        self._handles = []

    @staticmethod
    def _eligible(name: str) -> bool:
        if "lm_head" in name or "visual" in name:
            return False
        return "layers." in name

    def _hook(self, module, inputs):
        del module
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return None
        quantized = dynamic_symmetric_quantize(inputs[0], self.bits)
        self.stats.calls += 1
        self.stats.values += inputs[0].numel()
        return (quantized, *inputs[1:])

    def install(self, model) -> ActivationQuantStats:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and self._eligible(name):
                self._handles.append(module.register_forward_pre_hook(self._hook))
        self.stats.modules = len(self._handles)
        return self.stats

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


