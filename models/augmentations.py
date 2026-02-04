# import kornia.augmentation as K

# class Augmentations(nn.Module):
#     def __init__(self, augmentations: list = []):
#         super().__init__()
#         self.augmentations = augmentations

#     def forward(self, x):
#         for augmentation in self.augmentations:
#             x = augmentation(x)
#         return x