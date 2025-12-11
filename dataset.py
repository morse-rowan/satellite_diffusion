import os
import json
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class SatelliteDataset(Dataset):

    def __init__(self, root_dir, metadata_file, cloud_threshold=100.0, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cloud_threshold = cloud_threshold
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
            
        # filter based on cloud coverage
        self.filtered_data = [
            item for item in self.metadata 
            if item.get('cloud_coverage', 100.0) < self.cloud_threshold
        ]
        
        print(f"Loaded {len(self.filtered_data)} images after filtering for cloud coverage < {self.cloud_threshold}%")

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        item = self.filtered_data[idx]
        img_name = item['path']
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # return a black image or handle error appropriately
            raise e

        if self.transform:
            image = self.transform(image)
            
        return image

def get_data_loader(data_path, metadata_path, opts):
    """create training data loader."""
    
    # define transforms
    if opts.data_preprocess == "resize_only":
        train_transform = transforms.Compose([
            transforms.Resize(opts.image_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif opts.data_preprocess == "vanilla":
        load_size = int(1.1 * opts.image_size)
        osize = [load_size, load_size]
        train_transform = transforms.Compose([
            transforms.Resize(osize, Image.BICUBIC),
            transforms.RandomCrop(opts.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif opts.data_preprocess == 'deluxe':
        load_size = int(1.1 * opts.image_size)
        osize = [load_size, load_size]
        train_transform = transforms.Compose([
            transforms.Resize(osize, Image.BICUBIC),
            transforms.RandomCrop(opts.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(opts.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    dataset = SatelliteDataset(
        root_dir=data_path,
        metadata_file=metadata_path,
        cloud_threshold=opts.cloud_threshold,
        transform=train_transform
    )
    
    dloader = DataLoader(
        dataset=dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
    )

    return dloader
