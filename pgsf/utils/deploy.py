"""Reversible inference preparation: merge Conv/BN in a model copy."""
import copy
import torch
from torch import nn


def fuse_for_inference(model):
    fused=copy.deepcopy(model).eval()
    for module in list(fused.modules()):
        if hasattr(module,'conv') and hasattr(module,'bn') and hasattr(module,'forward_fuse'):
            if isinstance(module.conv,nn.Conv2d) and isinstance(module.bn,nn.BatchNorm2d):
                module.conv=torch.nn.utils.fuse_conv_bn_eval(module.conv,module.bn)
                delattr(module,'bn');module.forward=module.forward_fuse
        elif isinstance(module,nn.Sequential):
            for i in range(len(module)-1):
                if isinstance(module[i],nn.Conv2d) and isinstance(module[i+1],nn.BatchNorm2d):
                    module[i]=torch.nn.utils.fuse_conv_bn_eval(module[i],module[i+1])
                    module[i+1]=nn.Identity()
    return fused


class CudaGraphForward:
    """Capture an eval forward at fixed shapes to reduce CPU launch overhead.

    Inputs are copied into owned static buffers. Returned tensors share graph
    storage and are overwritten on the next call: consume or clone them first.
    The model must remain in eval mode with unchanged weights and fusion flags.
    Decode/NMS stays outside the graph because its output size is variable.
    """

    def __init__(self, model, example, warmup=3, decode=False):
        if model.training:
            raise ValueError('CUDA graph capture requires an eval model')
        if any(not isinstance(v, torch.Tensor) or not v.is_cuda for v in example.values()):
            raise ValueError('CUDA graph inputs must all be CUDA tensors')
        if any(k.startswith(('boxes_', 'centers_', 'mask')) for k in example):
            raise ValueError('Supply inference images/metadata only, without annotations')
        self.inputs = {k: v.clone() for k, v in example.items()}
        self.device = next(iter(example.values())).device
        self.graph = torch.cuda.CUDAGraph()
        def run():
            out=model(self.inputs)
            if decode:
                from pgsf.models.detector import decode_modality
                for name in ('rgb','tir'):
                    out['_decoded_'+name]=decode_modality(
                        out['pred_'+name],out['sel_'+name],self.inputs[name+'_hw'],
                        model.cfg.model)
            return out
        with torch.cuda.device(self.device), torch.inference_mode(), \
                torch.amp.autocast('cuda', dtype=torch.float16):
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    run()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            with torch.cuda.graph(self.graph):
                self.outputs = run()

    def __call__(self, batch):
        if set(batch) != set(self.inputs):
            raise ValueError('CUDA graph input keys differ from captured keys')
        for key, value in batch.items():
            static = self.inputs[key]
            if value.shape != static.shape or value.dtype != static.dtype or value.device != static.device:
                raise ValueError(f'CUDA graph shape/dtype/device changed for {key}')
            static.copy_(value)
        self.graph.replay()
        return self.outputs
