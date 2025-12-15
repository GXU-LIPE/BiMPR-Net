



![](figs/mmpc_arch.png)

## Introduction
With the continuous advancement of 3D visual technology, image-guided point cloud completion, as an emerging field within point cloud completion, has increasingly drawn the attention of researchers. Despite the substantial improvements in point cloud completion achieved by existing methods, challenges persist, including sensitivity to image viewpoints, difficulty in effectively handling cross-modal data, and the lack of finegrained semantic structure in the generated object point clouds. In this work, we introduce a novel cross-modal image-guided point cloud completion network named BiMPR-Net. Different from previous approaches to cross-modal data processing, BiMPR-Net employs a proposed bidirectional interactive compensation module. This module leverages multimodal information to bridge the information gap between point clouds and image modalities. By utilizing visual information from view images and geometric cues from the projected depth map, BiMPR-Net accurately predicts enhanced semantic and geometric feature representations of incomplete point clouds. We design a novel multi-scale progressive refinement module for refining point clouds, facilitating the effective fusion of global shapes and local details through dual-stream features. Ultimately, this approach achieves high-quality point cloud generation. Extensive quantitative and qualitative experiments conducted on the ShapeNet-ViPC benchmark dataset demonstrate that our image-guided point cloud completion method BiMPR-Net achieves state-of-the-art performance.

## Requirements
The code has been developed with the following dependecies:

- Python 3.8 
- CUDA version 10.2
- G++ or GCC 7.5.0
- Pytorch 1.10.2

To setup the environment and install all the required packages run:

```setup
sh setup.sh
```

It automatically creates the environment and install all the required packages.

If something goes wrong please consider to follow the steps in setup manually.



## Dataset 

The dataset is borrowed from ["View-guided point cloud completion"](https://github.com/Hydrogenion/ViPC).



## Training
The file config.py contains the configuration for all the training parameters.

To train the models in the paper, run this command:

```train
python train/train.py 
```


## Evaluation

To evaluate the models (select the specific category in config.py):

```eval
python eval/eval.py 
```





## Results

### [Point Cloud Completion on ShapeNet-ViPC](https://paperswithcode.com/sota/point-cloud-completion-on-shapenet-vipc)


<img src="figs/Visualization.jpg"  width="800" height="600">

## Acknowledgements
Some of the code is borrowed from [AXform](https://github.com/kaiyizhang/AXform) and [XMFnet](https://github.com/diegovalsesia/XMFnet). 

Visualizations have been created using [Mitsuba 2](https://www.mitsuba-renderer.org/).



## License 
Our code is released under MIT License (see LICENSE file for details).



