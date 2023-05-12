import numpy as np
import matplotlib.pyplot as plt
import torchvision

def feature_imshow(inp, title=None):
    """Imshow for Tensor.
    Reference: https: // www.zhihu.com / question / 68384370 / answer / 812588336
    """

    inp = inp.cpu().detach().numpy().transpose((1, 2, 0))

    mean = np.array([0.5, 0.5, 0.5])

    std = np.array([0.5, 0.5, 0.5])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig(title)





