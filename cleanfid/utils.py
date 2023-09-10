import numpy as np
import torch
import torchvision
from PIL import Image
from cleanfid.resize import build_resizer
import zipfile


class ResizeWrapperDataset(torch.utils.data.Dataset):
    """
    A wrapper around a dataset
    """

    def __init__(self, dataset, mode, size=(299, 299)):
        super().__init__()
        self.base_dataset = dataset
        self.transforms = torchvision.transforms.ToTensor()
        self.fn_resize = build_resizer(mode)
        self.custom_image_tranform = lambda x: x

    def __len__(self):
        return len(self.bae_dataset)

    def __getitem__(self, i):
        elem_i = self.base_dataset[i]

        # elem_i is expected to be either a tuple of 1 or 2 elements
        # dependening on whether or not the dataset is labeled
        if isinstance(elem_i, tuple):
            if len(elem_i) == 1:
                input_i = elem_i[0]
            elif len(elem_i) == 2:
                input_i, _ = elem_i
            else:
                raise RuntimeError(
                    "A mappable dataset returning more than one element is not handled"
                )
        else:
            input_i = elem_i

        # The dataset is expected to return a tensor (C, H, W)
        # with values in [0, 1], e.g. MNIST dataset with transforms.ToTensor
        if not isinstance(input_i, torch.Tensor):
            raise RuntimeError(
                f"Expected the dataset to return a torch.Tensor, got {type(input_i)} instead"
            )
        if len(input_i.shape) != 3:
            raise RuntimeError(
                f"Expected the input tensor to be 3-dimensional, got {len(input_i.shape)} dimensional tensor instead"
            )

        # Convert the pytorch tensor to a nd array in [0, 255]
        # The resizer is expecting a nd array in this range
        np_input_i = 255 * input_i.numpy()
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(np_input_i)
        img_t = self.transforms(img_resized)

        return img_t


class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=(299, 299), fdir=None):
        self.files = files
        self.fdir = fdir
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = build_resizer(mode)
        self.custom_image_tranform = lambda x: x
        self._zipfile = None

    def _get_zipfile(self):
        assert self.fdir is not None and ".zip" in self.fdir
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.fdir)
        return self._zipfile

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = str(self.files[i])
        if self.fdir is not None and ".zip" in self.fdir:
            with self._get_zipfile().open(path, "r") as f:
                img_np = np.array(Image.open(f).convert("RGB"))
        elif ".npy" in path:
            img_np = np.load(path)
        else:
            img_pil = Image.open(path).convert("RGB")
            img_np = np.array(img_pil)

        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized)) * 255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t


EXTENSIONS = {
    "bmp",
    "jpg",
    "jpeg",
    "pgm",
    "png",
    "ppm",
    "tif",
    "tiff",
    "webp",
    "npy",
    "JPEG",
    "JPG",
    "PNG",
}
