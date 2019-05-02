import torch
from torch import nn
from torchvision.models import resnet18
from PIL import Image
import numpy as np


class GReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input = grad_input.clamp(min=0)
        return grad_input


class GuidedReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return GReLU.apply(x)


class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.replace(model)

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


if __name__ == "__main__":
    from matplotlib.pyplot import imshow,figure, colorbar
    m = resnet18(pretrained=True)
    m.eval()
    g = GuidedBackprop(m)
    x = np.array(Image.open('cat.jpg'))
    # imshow(x)
    x = x.astype('f') / 255
    x -= np.array([0.485, 0.456, 0.406])
    x /= np.array([0.229, 0.224, 0.225])
    x = torch.from_numpy(x.transpose(2, 0, 1)[None, ...])
    d = g.probe(x, 242)
    d[d < 0] = 0
    d = d.sum((0, 1)).numpy()
    d = (d - d.min())/(d.max() - d.min()) * 255
    figure()
    imshow(d.astype('i'))
    colorbar()