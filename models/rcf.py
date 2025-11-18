from torchgeo.models import RCF
import torch.nn as nn
import torch
import numpy as np
from kornia.enhance import normalize
from pathlib import Path
import wandb
from tqdm import tqdm
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule



class NaturalnessDataset(Dataset):
    def __init__(self, rhs_fp: str = None, target_df: pd.DataFrame = None, use_full_profile: bool = False, transform=None):
        super().__init__()
        self.rhs_fp = Path(rhs_fp).expanduser()
        self.target_df = target_df
        self.transform = transform
        if use_full_profile:
            self.idx = slice(None)
        else:
            self.idx = slice(98, 99)
        
    def __len__(self):
        if not hasattr(self, 'data'):
            self.data = h5py.File(self.rhs_fp)
        return len(self.data['rhs_median'])
    
    def __getitem__(self, idx):
        if not hasattr(self, 'data'):
            self.data = h5py.File(self.rhs_fp)
        rhs = self.data['rhs_median'][idx, self.idx] # [101, 15, 15]
        s2 = self.data['s2'][idx] # [12, 15, 15]
        rowid = self.data['rowid'][idx]
        # Return the mapped class index instead of original land use ID
        target = self.target_df.loc[rowid, 'class_idx']
        return rhs, s2, target
    
    def __del__(self):
        if hasattr(self, 'data'):
            self.data.close()
            

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        return normalize(x.float(), self.mean, self.std)

class NaturalnessDataModule(LightningDataModule):
    def __init__(self, h5_file: str = None, naturalness_fp: str = None, use_full_profile: bool = False, 
                 batch_size: int = 1, num_workers: int = 4, train_val_split: float = 0.8, 
                 class_balance: bool = False,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Map original land use IDs to consecutive class indices
        self.land_use_mapping = {
            0: 0,   # No forest
            11: 1,  # Naturally regenerating forest without any signs of human activities, e.g., primary forests.
            20: 2,  # Naturally regenerating forest with signs of human activities, e.g., logging, clear cuts etc.  
            31: 3,  # Planted forest.
            32: 4,  # Short rotation plantations for timber.  
            40: 5,  # Oil palm plantations.  
            53: 6   # Agroforestry. 
        }
        
        self.target_df = pd.read_csv(naturalness_fp, index_col='rowid')
        # remove unsure labels (-1) and labels with no high resolution images (1)
        self.target_df = self.target_df[~self.target_df['Land_use_ID'].isin([1, -1])]
        self.target_df['class_idx'] = self.target_df['Land_use_ID'].map(self.land_use_mapping)
        # Fill NaN values with 0 (Unknown/No data class)
        self.target_df['class_idx'] = self.target_df['class_idx'].fillna(0)
        self.target_df['class_idx'] = self.target_df['class_idx'].astype(int)
        self.target_df = self.target_df.astype({'class_idx': 'int64'})

        self.full_dataset = NaturalnessDataset(h5_file, self.target_df, use_full_profile=use_full_profile)

        # Create reverse mapping for reference
        self.id_to_land_use = {v: k for k, v in self.land_use_mapping.items()}
        if class_balance:
            class_counts = self.target_df['class_idx'].value_counts()
            min_count = class_counts.min()
            self.target_df = self.target_df.groupby('class_idx', group_keys=False).apply(lambda x: x.sample(min_count, random_state=42))
        
        self.target_df_train = self.target_df.groupby('class_idx', group_keys=False).apply(lambda x: x.sample(frac=0.8))
        self.target_df_val = self.target_df[~self.target_df.index.isin(self.target_df_train.index)]
        with h5py.File(Path(h5_file).expanduser()) as f:
            rowids = f['rowid'][:]
        self.train_idx = np.where(np.isin(rowids, self.target_df_train.index))[0]
        self.val_idx = np.where(np.isin(rowids, self.target_df_val.index))[0]
        self.train_dataset = torch.utils.data.Subset(self.full_dataset, self.train_idx)
        self.val_dataset = torch.utils.data.Subset(self.full_dataset, self.val_idx)
         
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, shuffle=False, drop_last=False)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         num_workers=self.num_workers, shuffle=False, drop_last=False)

class RCFNaturalness(RCF):
    def __init__(self, in_channels: int, features: int, kernel_size: int, bias: float = -1.0,
                 dataset: NaturalnessDataset = None, mode: str = 'empirical', seed: int = 42):
        super().__init__(in_channels=in_channels, features=features, kernel_size=kernel_size,
                         bias=bias, dataset=dataset, mode='gaussian', seed=seed)
        assert features % 2 == 0
        num_patches = features // 2

        generator = torch.Generator()
        if seed:
            generator = generator.manual_seed(seed)

        if mode == 'empirical':
            assert dataset is not None
            num_channels, height, width = dataset[0][0].shape # changed here
            assert num_channels == in_channels
            patches = np.zeros(
                (num_patches, num_channels, kernel_size, kernel_size), dtype=np.float32
            )
            idxs = torch.randint(
                0, len(dataset), (num_patches,), generator=generator
            ).numpy()
            ys = torch.randint(
                0, height - kernel_size, (num_patches,), generator=generator
            ).numpy()
            xs = torch.randint(
                0, width - kernel_size, (num_patches,), generator=generator
            ).numpy()

            for i in range(num_patches):
                img = dataset[idxs[i]][0] # changed here
                patches[i] = img[
                    :, ys[i]: ys[i] + kernel_size, xs[i]: xs[i] + kernel_size
                ]

            patches = self._normalize(patches)
            self.weights = torch.tensor(patches)

        self.classifier = nn.Linear(features, 7)

    def forward(self, x):
        # x (batch_size, 101, 15, 15)
        x = super().forward(x) # x (batch_size, 4096)
        x = self.classifier(x) # x (batch_size, 7)
        return x


dataloader = NaturalnessDataModule(
    h5_file='/projects/dereeco/data/gvs/downstream_task_data/rhs_predictions_2017_0crmfaia_ps31.h5', # /projects/dereeco/data/...
    naturalness_fp='/projects/dereeco/data/gvs/downstream_task_data/naturalness/reference_data_set_updated.with_images.csv',
    use_full_profile=True, batch_size=512)
# dataset = copy.deepcopy(dataloader.train_dataset)
rcf_naturalness = RCFNaturalness(in_channels=101, features=4096, kernel_size=3, dataset=dataloader.train_dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
rcf_naturalness.to(device)
mean_std_fp = Path("/projects/dereeco/data/gvs/downstream_task_data/naturalness/mean_std_0crmfaia.npz").expanduser()
mean_std = np.load(mean_std_fp)
mean = mean_std['mean']
std = mean_std['std']
optimizer = torch.optim.Adam(rcf_naturalness.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
max_epochs = 200
wandb.init(project='global-vegetation-structure-v1', name='rcf-naturalness')
for epoch in range(max_epochs):
    train_loss = 0
    train_accuracy = 0
    val_loss = 0
    val_accuracy = 0
    rcf_naturalness.train()
    for batch in tqdm(dataloader.train_dataloader(), desc='Training'):
        rhs, _, target = batch
        rhs = rhs.to(device)
        target = target.to(device)
        x = normalize(rhs.float(), mean, std)
        x = rcf_naturalness(x)
        loss = loss_fn(x, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        accuracy = (x.argmax(dim=1) == target).float().mean()
        train_loss += loss.item()
        train_accuracy += accuracy.item()
    rcf_naturalness.eval()
    for batch in tqdm(dataloader.val_dataloader(), desc='Validating'):
        rhs, _, target = batch
        rhs = rhs.to(device)
        target = target.to(device)
        x = normalize(rhs.float(), mean, std)
        x = rcf_naturalness(x)
        loss = loss_fn(x, target)
        accuracy = (x.argmax(dim=1) == target).float().mean()
        val_loss += loss.item()
        val_accuracy += accuracy.item()
    wandb.log({'epoch': epoch, 'train.loss': train_loss / len(dataloader.train_dataloader()),
               'acc/train': train_accuracy / len(dataloader.train_dataloader()),
               'val.loss': val_loss / len(dataloader.val_dataloader()),
               'acc/val': val_accuracy / len(dataloader.val_dataloader())})
