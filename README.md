# Contrastive General Graph Matching with Adaptive Augmentation Sampling
Official implementation of IJCAI'24 paper [Contrastive General Graph Matching with Adaptive Augmentation Sampling](https://arxiv.org/abs/2406.17199)

## Setup (Ubuntu)

The codebase is verified on Ubuntu 16.04 with Python 3.7, Pytorch 1.6, CUDA 10.1, CUDNN 7, and torch-geometric 1.6.3.

1. Set up Pytorch 1.6 with support for GPU.

2. Install ninja-build with: `apt-get install ninja-build`

3. Add Python libraries:

   ```bash
   pip install tensorboardX scipy easydict pyyaml xlrd xlwt pynvml pygmtools wandb
   ```

4. Prepare LPMP build essentials:

   ```bash
   apt-get install -y findutils libhdf5-serial-dev git wget libssl-dev

   wget https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1.tar.gz && tar zxvf cmake-3.19.1.tar.gz
   cd cmake-3.19.1 && ./bootstrap && make && make install
   ```

5. Download and compile LPMP:

   ```bash
   python -m pip install git+https://git@github.com/rogerwwww/lpmp.git
   ```

   For a successful LPMP build, `gcc-9` may be necessary. To install and set up `gcc-9`, follow this example:

   ```bash
   apt-get update
   apt-get install -y software-properties-common
   add-apt-repository ppa:ubuntu-toolchain-r/test

   apt-get install -y gcc-9 g++-9
   update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
   ```

6. Install torch-geometric by running:

   ```bash
   export CUDA=cu101
   export TORCH=1.6.0
   /opt/conda/bin/pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
   /opt/conda/bin/pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
   /opt/conda/bin/pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
   /opt/conda/bin/pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
   /opt/conda/bin/pip install torch-geometric==1.6.3
   ```

7. If `gcc-9` was used for LPMP, revert to `gcc-7` afterwards, as shown below:

   ```bash
   update-alternatives --remove gcc /usr/bin/gcc-9
   update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
   ```

8. If there was an error related to the package, please check `requirements.txt` for the correct version.

## Datasets

The datasets below are obtainable through `pygmtools` automatically or manual download in case of any download issues.

1. PascalVOC-Keypoint

   - Get the [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and ensure to have it as: `data/PascalVOC/TrainVal/VOCdevkit/VOC2011`
   - Retrieve keypoint annotations for VOC2011 from the [Berkeley server](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) or [google drive](https://drive.google.com/open?id=1D5o8rmnY1-DaDrgAXSygnflX5c-JyUWR) to ensure they looks like: `data/PascalVOC/annotations`
   - The train/test split is available in ``data/PascalVOC/voc2011_pairs.npz``

2. SPair-71k

   - Obtain the dataset from [this link](http://cvlab.postech.ac.kr/research/SPair-71k/).


   - Ensure images are placed in `data/SPair-71k/JPEGImages`, and annotations in `data/SPair-71k/ImageAnnotation`.

3. Willow ObjectClass

   - The Willow dataset can be found on [this page](http://www.di.ens.fr/willow/research/graphlearning/).


- After downloading, unzip and store images and annotations are in `data/WillowObject/WILLOW-ObjectClass`

4. Synthetic Dataset

   Due to the size limitations of the supplementary material, the synthetic dataset was designed to be automatically generated and saved during the execution of the experiments.

   ```bash
   python train_eval_gcgm.py --cfg experiments/Synthetic/GCGM/synthetic_gcgm.yaml
   ```

## Training and Validation Splits

To simplify the implementation process, we perform training and validation splits dynamically, controlled by random seeds. The five random seeds used to report the experimental results are `0, 1, 2, 3, 123`. Please update these values in the configuration file, under the `RANDOM_SEED` field.

## Run the Experiment

Training and evaluation GCGM:

```bash
python train_eval_gcgm.py --cfg experiments/GCGM/willow_gcgm_ngmv2.yaml
```

Training and evaluation other baselines:

```bash
python train_eval_baseline.py --cfg experiments/SCGM/willow_scgm_ngmv2.yaml
```