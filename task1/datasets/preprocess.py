import numpy as np
from tqdm import tqdm
from torchvision import transforms
from skimage import io, transform, color

class Preprocess(object):
    def __init__(self, cfg):
        self.background = BackGround(
            background=cfg['back']['background'],
            img_size=cfg['back']['img_size']
        )
        self.normalize = transforms.Normalize(
            mean=cfg['norm']['mean'],
            std=cfg['norm']['std']
        )
        self.to_tensor = transforms.ToTensor()
        
    def __call__(self, dataset):

        for data in tqdm(dataset, leave=False, desc='preprocessing') :
            img = data["image"]
            img = self.background(img)
            img = self.to_tensor(img)
            img = self.normalize(img)
            img = img.float()
            data["image"] = img
            
        return dataset
        

class BackGround(object):
    """Operator that resizes to the desired size while maintaining the ratio
        fills the remaining part with a black background

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size.
    """

    def __init__(self, background, img_size):
        self.background = background
        self.img_size = img_size

    def __call__(self, img):
        height, width, _ = img.shape

        new_img = np.zeros((self.img_size, self.img_size, 3))
        new_img += self.background

        if height != width :
            if width == self.img_size :
                height_start = int((self.img_size - height) / 2)
                new_img[height_start:height_start+height, :, :] = img
            else :
                width_start = int((self.img_size - width) / 2)
                new_img[:, width_start:width_start+width, :] = img
        else :
            new_img = img

        return new_img


