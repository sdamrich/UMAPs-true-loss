import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset
import os


path = "cifar10"


# transforms from URL that are expected by pretrained ResNet
transforms = transforms.Compose([#transforms.ToPILImage(), # perhaps omit
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225]),
                                ])

# get CIFAR10 datasets
ds_train = torchvision.datasets.CIFAR10(root=path,
                                        train=True,
                                        transform=transforms,
                                        download=True)
ds_test = torchvision.datasets.CIFAR10(root=path,
                                       train=False,
                                       transform=transforms,
                                       download=True)

ds = torch.utils.data.ConcatDataset([ds_train, ds_test])
print(f"Length of dataset: {len(ds)}")

dl = torch.utils.data.DataLoader(ds,
                                 batch_size=32,
                                 num_workers=0,
                                 shuffle=False,
                                 drop_last=False)


# get the pretrained ResNet
torch.hub.set_dir(path)
resnet50 = torchvision.models.resnet50(pretrained=True, progress=True).cuda().eval()
feature_extractor = torch.nn.Sequential(resnet50.conv1,
                                        resnet50.bn1,
                                        resnet50.relu,
                                        resnet50.maxpool,
                                        resnet50.layer1,
                                        resnet50.layer2,
                                        resnet50.layer3,
                                        resnet50.layer4,
                                        resnet50.avgpool)


# extract features
features = []
labels = []
print("Extracting features")
with torch.no_grad():
    for batch in tqdm(dl):
        features.append(feature_extractor(batch[0].cuda()).data.cpu().detach())
        labels.append(batch[1])

features = torch.cat(features).reshape(len(ds), -1).numpy()
labels = torch.cat(labels).numpy()

# save features and labels
print("Saving features")
np.save(os.path.join(path, f"cifar10_resnet50_features.npy"), features)
np.save(os.path.join(path, "cifar10_labels.npy"), labels)
