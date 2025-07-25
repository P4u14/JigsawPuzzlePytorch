from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=(256, 256)):
        """
        Args:
            image_paths (list of str): List of paths to input images.
            mask_paths (list of str): List of paths to corresponding masks.
            img_size (tuple): Size to which images and masks will be resized.
        """
        self.images = image_paths
        self.masks = mask_paths
        self.transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        img = self.transform(img)
        mask = (self.transform(mask) > 0.5).float()
        return img, mask