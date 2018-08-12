# Apolloscape dataset for localization task.
Exploring localization task on [Apolloscape](http://apolloscape.auto/) dataset.

ECCV2018 Self-localization on-the-fly task - [challenge page](http://apolloscape.auto/ECCV/challenge.html).

__NOTE: This repository is a work in progress.__

# Prerequisites

Dataset reader based on Pytorch 0.4.0 `Dataset`. To install all dependencies:
```
pip install -r requirements.txt
```

# Data

Download data from Apolloscape [page](http://apolloscape.auto/scene.html) and unpack it to a folder. Examples below assume that data folder symbolically linked to `./data/apolloscape`.

```
ln -s <DATA FOLDER>/apolloscape ./data
```

Sample data file for `zpark` road provided in localization challenge section supported automatically (it has different folder names, files order and pose data files format)

# Python Notebook example

See roads and record graphs in [Apolloscape_View_Records Notebook](./Apolloscape_View_Records.ipynb)

PoseNet training, error calculation and result visualization in [Apolloscape_PoseNet](./Apolloscape_PoseNet.ipynb)

![PoseNet on Train](./assets/posenet_2048_train.png)

# Show/Save path and sample images by record id

```
python plot_dataset.py --data ./data/apolloscape --road road03_seg --record Record018
```

![Record path](./assets/road03_seg_Record018_00267.png)

# Generate video of the path by record id

```
python plot_dataset.py --data ./data/apolloscape --road road03_seg --record Record018 --video
```

# Train PoseNet on ZPark road

```
python train.py --data ./data/apolloscape --road zpark-sample --checkpoint-save 10 --fig-save 1 --epochs 50 --lr 1e-5 --experiment zpark_posenet_L1 --feature-net resnet34 --feature-net-pretrained
```



![Record video](./assets/road03_seg_Record018.gif)

# TODO:
* implement `stereo=False` mode and train on not filtered data
* PoseNet with automatic weights learning
* VidLoc implementation
* [Optional] Prepare data for [eval script](https://github.com/ApolloScapeAuto/dataset-api/tree/master/self_localization)
* SfM / 3D Reconstruction pipeline
* WGAN for generating new samples
* SfM for 3D map
* Qt/OpenGL visualizations
* Test GeoMapNet and other recent solutions
