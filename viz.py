import torch
from torch import nn
from torch.nn.functional import interpolate
from copy import deepcopy


class GReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input * grad_output < 0.02] = 0
        # grad_input = grad_input.clamp(min=0)

        return grad_input


# class GSigmoid(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input*(1-input))
#         return input.sigmoid()
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         derivative, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input = grad_input.clamp(min=0)
#         grad_input *= derivative
#         return grad_input


class GuidedReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return GReLU.apply(x)


class GuidedBackprop:
    def __init__(self, model):
        self.model = deepcopy(model).eval()
        self.replace(self.model)

    def replace(self, container):
        for key, mod in container._modules.items():
            if isinstance(mod, nn.ReLU):
                container._modules[key] = GuidedReLU()
            if mod._modules:
                self.replace(mod)

    def probe(self, x, target_class):
        if isinstance(target_class, int):
            target_class = (target_class,)
        x.requires_grad_()
        x.grad = None
        out = self.model(x)
        out[(slice(0, None),)+target_class].backward()
        return x.grad


class GradCAM:
    def __init__(self, model, layer_nums):
        self.model = model.eval()
        self.saved_activation = []
        self.hook(model, layer_nums)

    def hook(self, container, layer_nums, cnt=0):

        def forward_hook(module, input, output):
            output.retain_grad()
            self.saved_activation.append(output)

        if isinstance(layer_nums, int):
            layer_nums = [layer_nums]

        for key, mod in container._modules.items():
            if mod._modules:
                cnt = self.hook(mod, layer_nums, cnt)
            elif cnt in layer_nums:
                mod.register_forward_hook(forward_hook)
                print(key, mod, cnt)
                cnt += 1
            else:
                cnt += 1
        return cnt

    def probe(self, x, target_class):
        self.saved_activation = []
        if isinstance(target_class, int):
            target_class = (target_class,)
        out = self.model(x)
        self.model.zero_grad()
        out[(slice(0, None),)+target_class].backward()
        heatmaps = []
        for act in self.saved_activation:
            heatmaps.append(interpolate(torch.einsum('ab,abcd->acd', act.grad.sum((-2, -1)), act).unsqueeze(1),
                            size=tuple(x.shape[-2:]), mode='bilinear'))
        return heatmaps
