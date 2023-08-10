import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform, color

class BBoxCrop(object):

    def __call__(self, image, x_1, y_1, x_2, y_2):
        h, w = image.shape[:2]

        top = y_1
        left = x_1
        new_h = y_2 - y_1
        new_w = x_2 - x_1

        image = image[top: top + new_h,
                      left: left + new_w]

        return image

class CropResize(object):
    
    def __call__(self, image, img_size):
        h, w = image.shape[:2]

        if isinstance(img_size, int):
            if h > w:
                new_h, new_w = img_size, img_size * w / h   
            else:
                new_h, new_w = img_size * h / w, img_size   
        else:
            new_h, new_w = img_size
        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w), mode='constant')

        return image

class ETRIDataset_emo(Dataset):
    
    def __init__(self, df, img_size, base_path):
        self.df = df
        self.img_size = img_size
        self.base_path = base_path
        self.bbox_crop = BBoxCrop()
        self.crop_resize = CropResize()
      
    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = io.imread(self.base_path + row['image_name'])
        if image.shape[2] != 3:
            image = color.rgba2rgb(image)

        bbox_xmin = row['BBox_xmin']
        bbox_ymin = row['BBox_ymin']
        bbox_xmax = row['BBox_xmax']
        bbox_ymax = row['BBox_ymax']

        image = self.bbox_crop(image, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
        image = self.crop_resize(image, self.img_size)

        color = np.zeros(len(self.df["Color"].unique()))

        color[row["Color"]] = 1.0

        ret = {}
        ret['image'] = image
        ret['color'] = color

        return ret

    def __len__(self):
        return len(self.df)

class Datasets(object):

    def __call__(self, df, img_size, base_path):
        dataset = []
        get_data = ETRIDataset_emo(df, img_size, base_path)
        for data in tqdm(get_data, leave=False, desc='dataset loading'):
            dataset.append(data)
        random.shuffle(dataset)
        return dataset