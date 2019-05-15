from matplotlib.pyplot import imshow, figure  # , colorbar
from .viz import GuidedBackprop, read_im
from torchvision.models import resnet18


def gen_fig(d):
    d[d < 0] = 0
    d = d = d[0].numpy()
    figure()
    if d.sum() != 0:
        d = (d - d.min()) / (d.max() - d.min()) * 255
    imshow(d.transpose(1, 2, 0).astype('i'))


if __name__ == "__main__":
    m = resnet18(pretrained=True)
    g = GuidedBackprop(m)
    # g = GradCAM(m, 48)
    x = read_im("cat_dog.png")
    gen_fig(g.probe(x, 281))
    gen_fig(g.probe(x, 242))
    # imshow(d[0,0].detach().numpy())
    # colorbar()