from matplotlib.pyplot import imshow, figure  # , colorbar
from viz import GuidedBackprop, GradCAM
import torch
from torchvision.models import resnet18
import numpy as np
from PIL import Image


class FGSM:
    def __init__(self, model, epsilon=1e-2):
        self.model = model.eval()
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
            x = scale_transform(x - self.epsilon * torch.sign(x.grad) * mask.float()).detach()
            # self.show(x)
            x.requires_grad_()
            cur_class = self.model(x).argmax(1)
            print(cur_class)
            mask = cur_class == orig_class
        self.show(x)
        return x, cur_class


def read_im(im_name, show=True):
    from matplotlib.pyplot import imshow, figure
    x = np.array(Image.open(im_name))
    if show:
        figure()
        imshow(x)
    x = scale_transform(x.astype('f'))
    return torch.from_numpy(x.transpose(2, 0, 1)[None, ...])


def scale_transform(x):
    x = (x - x.min())/(x.max() - x.min())
    if isinstance(x, np.ndarray):
        x -= np.array([0.485, 0.456, 0.406])
        x /= np.array([0.229, 0.224, 0.225])
    elif isinstance(x, torch.Tensor):
        x -= torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        x /= torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return x

def gen_fig(d):
    d[d < 0] = 0
    d = d[0].detach().numpy()
    figure()
    if d.sum() != 0:
        d = (d - d.min()) / (d.max() - d.min()) * 255
    imshow(d.astype('i'))


if __name__ == "__main__":
    m = resnet18(pretrained=True)
    # g = GuidedBackprop(m)
    g = GradCAM(m, 48)
    x = read_im("cat_dog.png")
    gen_fig(g.probe(x, 281)[0][0])
    gen_fig(g.probe(x, 242)[0][0])
    # imshow(d[0,0].detach().numpy())
    # colorbar()