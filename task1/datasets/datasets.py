import multiprocessing
import parmap
import random
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class BBoxCrop(object):

    def __call__(self, image, x_1, y_1, x_2, y_2):
        _, h, w = image.shape
        image = image[:, y_1: y_2, x_1: x_2]
        return image

class CropResize(object):
    
    def __call__(self, image, img_size):
        _, h, w = image.shape
        if isinstance(img_size, int):
            if h > w:
                new_h, new_w = img_size, img_size * w / h   
            else:
                new_h, new_w = img_size * h / w, img_size   
        else:
            new_h, new_w = img_size
        new_h, new_w = int(new_h), int(new_w)
        return transforms.Resize((new_h, new_w))(image)

class ETRIDataset_emo(Dataset):
    
    def __init__(self, df, img_size, base_path):
        self.df = df
        self.img_size = img_size
        self.base_path = base_path
        self.bbox_crop = BBoxCrop()
        self.crop_resize = CropResize()
      
    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = transforms.ToTensor()(
            Image.open(self.base_path + row['image_name']).convert('RGB'))

        bbox_xmin = row['BBox_xmin']
        bbox_ymin = row['BBox_ymin']
        bbox_xmax = row['BBox_xmax']
        bbox_ymax = row['BBox_ymax']

        image = self.bbox_crop(image, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
        image = self.crop_resize(image, self.img_size)

        daily = torch.zeros(len(self.df["Daily"].unique()))
        gender = torch.zeros(len(self.df["Gender"].unique()))
        embel = torch.zeros(len(self.df["Embellishment"].unique()))

        daily[row["Daily"]] = 1.0
        gender[row["Gender"]] = 1.0
        embel[row["Embellishment"]] = 1.0

        ret = {}
        ret['image'] = image
        ret['daily'] = daily
        ret['gender'] = gender
        ret['embel'] = embel

        return ret

    def __len__(self):
        return len(self.df)

class Datasets(object):
    def __init__(self, df, img_size, base_path):
        self.data = ETRIDataset_emo(df, img_size, base_path)
    
    def get_data(self, i):
        return self.data[i]

    def get_dataset(self):
        dataset = []
        for data in tqdm(self.data, leave=False, desc='dataset loading'):
            dataset.append(data)
        random.shuffle(dataset)
        return dataset      
    
    def get_dataset_parallel(self):
        num_cores = multiprocessing.cpu_count()
        dataset = parmap.map(self.get_data, 
            range(len(self.data)), 
            pm_pbar=True, 
            pm_processes=num_cores
        )
        random.shuffle(dataset)
        return dataset