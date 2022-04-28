import torch.utils.data as data
import numpy as np
import datetime

from os import listdir
from os.path import join, abspath
from .utils import load_nifti_img, check_exceptions, is_image_file


class LITS3DDataset(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=False):
        super(LITS3DDataset, self).__init__()
        image_dir = join(root_dir, split, 'image')
        target_dir = join(root_dir, split, 'segmentation')
        self.image_filenames  = sorted([abspath(join(image_dir, x)) for x in listdir(image_dir) if is_image_file(x)])
        self.target_filenames = sorted([abspath(join(target_dir, x)) for x in listdir(target_dir) if is_image_file(x)])
        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Number of {0} images: {1} NIFTIs'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [load_nifti_img(ii, dtype=np.int16)[0] for ii in self.image_filenames]
            self.raw_labels = [load_nifti_img(ii, dtype=np.uint8)[0] for ii in self.target_filenames]
            print('Loading is done\n')

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        # TODO: time-dependent seeds, that is not reproducible!
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            image, _ = load_nifti_img(self.image_filenames[index], dtype=np.int16)
            target, _ = load_nifti_img(self.target_filenames[index], dtype=np.uint8)
        else:
            image = np.copy(self.raw_images[index])
            target = np.copy(self.raw_labels[index])

        # handle exceptions
        check_exceptions(image, target)
        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.image_filenames)
