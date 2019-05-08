import torch
from torch import nn
from torchvision.models import resnet18
from PIL import Image
import numpy as np
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


class GSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input*(1-input))
        return input.sigmoid()

    @staticmethod
    def backward(ctx, grad_output):
        derivative, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = grad_input.clamp(min=0)
        grad_input *= derivative
        return grad_input


class GuidedReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return GReLU.apply(x)


class GuidedBackprop:
    def __init__(self, model):
        self.model = deepcopy(model)
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
        self.model = model
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


def read_im(im_name, show=True):
    from matplotlib.pyplot import imshow, figure
    x = np.array(Image.open(im_name))
    if show:
        figure()
        imshow(x)
    x = x.astype('f') / 255
    x -= np.array([0.485, 0.456, 0.406])
    x /= np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(x.transpose(2, 0, 1)[None, ...])


class FGSM:
    def __init__(self, model, epsilon=1e-2):
        self.model = model
        self.model.eval()
        self.epsilon = epsilon

    def show(self, x):
        from matplotlib.pyplot import imshow, figure
        figure()
        x = x.detach().numpy()[0].transpose(1, 2, 0)
        # print(x.shape)
        x = (x - x.min((0, 1)))/(x.max((0, 1), ) - x.min((0, 1))) * 255
        imshow(x.astype('i'))

    def adv_sample(self, x):
        x = x.detach().clone().requires_grad_()
        out = self.model(x)
        orig_class = out.argmax(1)
        cur_class = orig_class
        mask = cur_class == orig_class
        while mask.any():
            self.model(x)[:, orig_class].backward()
            x = x.detach() - self.epsilon * torch.sign(x.grad) * mask.float()
            # self.show(x)
            x.requires_grad_()
            cur_class = self.model(x).argmax(1)
            print(cur_class)
            mask = cur_class == orig_class
        self.show(x)
        return x, cur_class


if __name__ == "__main__":
    from matplotlib.pyplot import imshow,figure # , colorbar
    m = resnet18(pretrained=True)
    m.eval()
    g = GuidedBackprop(m)
    # g = GradCAM(m, 48)
    x = read_im("cat.jpg")
    d = g.probe(x, 281)
    d[d < 0] = 0
    d = d[0].numpy()
    d = (d - d.min())/(d.max() - d.min()) * 255
    figure()
    imshow(d.transpose(1, 2, 0).astype('i'))
    # imshow(d[0,0].detach().numpy())
    #colorbar()