import os
import time
import cv2
import numpy as np
from pathlib import Path
from mss import mss
from fastai.vision.all import *

def run():
    inputs = {}
    for data_folder in os.listdir("data"):
        inputs[str(data_folder)] = np.loadtxt("data/"+data_folder+"/inputs.txt")

    def label_func(x):
        name = str(x.parent.parent.name)
        inp = inputs[name]
        print(x)
        return inp[int(x.stem)]

    path = Path("data/")
    img_files = get_image_files(path)
    print(f'Found {len(img_files)} images')
    label0 = label_func(img_files[0])
    print("Label0:",label0)

    #data = ImageDataLoaders.from_path_func(path, fnames, label_func, bs=40, num_workers=0)
    # data = (ImageItemList.from_folder(path)
    #         .random_split_by_pct()
    #         .label_from_func(label_func, label_cls=FloatList)
    #         .transform(get_transforms(), size=224)
    #         .databunch())
    # data.normalize(imagenet_stats)
    biwi = DataBlock(
        blocks=(ImageBlock, RegressionBlock),
        get_items=get_image_files,
        get_y=label_func,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        batch_tfms=[*aug_transforms(size=(240, 320)),
                Normalize.from_stats(*imagenet_stats)]
        )

    dls = biwi.dataloaders(path, bs=5)
    xb, yb = dls.one_batch()
    dls.show_batch(max_n=2, figsize=(2,2))

    learn = cnn_learner(dls, resnet18, y_range=(-1,1))
    learn.loss = MSELossFlat
    print("Loaded")
    learn.fine_tune(4, base_lr=1.0e-02)

    os.mkdir("brains/")
    learn.path = Path("brains/")
    learn.export()

if __name__ == '__main__':
    run()
