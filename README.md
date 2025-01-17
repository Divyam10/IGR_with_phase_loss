# IGR: Implicit Geometric Regularization for Learning Shapes
<p align="center">
  <img src="IGR.png"/>
</p>

This repository contains an implementation to the ICML 2020 paper: "Implicit Geometric Regualrization for Learning Shapes".

IGR is a deep learning approach for learning implicit signed distance representations directly from raw point clouds with or without normal data.
Our method aims to find an SDF by optimizing the network to solve the eikonal equation with the input point cloud as boundary condition.
Although this is an ill posed condition we enjoy an implicit regualrization coming from the optimization procedure itself which aims our method to simple natrual solutions as can be seen on the figure above.

For more details:

paper: https://arxiv.org/abs/2002.10099.

video: https://youtu.be/6cOvBGBQF9g.


## Installation Requirmenets
The code is compatible with python 3.7 and pytorch 1.2. In addition, the following packages are required:  
numpy, pyhocon, plotly, scikit-image, trimesh.

## Usage


### Surface reconstruction
<p align="center">
  <img src="recon3D.png"/>
</p>

IGR can be used to reconstruct a single surface given a point cloud with or without normal data. Adjust reconstruction/setup.json to the
path of the input 2D/3D point cloud:
```
train
{
  ...
  d_in=D
  ...
  input_path = your_path
  ...
}
```
Where D=3 in case we use 3D data or 2 if we use 2D. We support xyz,npy,npz,ply files.

Then, run training:
```
cd ./code
python reconstruction/run.py 
```
Finally, to produce the meshed surface, run:
```
cd ./code
python reconstruction/run.py --eval --checkpoint CHECKPOINT
```
where CHECKPOINT is the epoch you wish to evaluate of 'latest' if you wish to take the most recent epoch.



## Citation
If you find our work useful in your research, please consider citing:

    @incollection{icml2020_2086,
     author = {Gropp, Amos and Yariv, Lior and Haim, Niv and Atzmon, Matan and Lipman, Yaron},
     booktitle = {Proceedings of Machine Learning and Systems 2020},
     pages = {3569--3579},
     title = {Implicit Geometric Regularization for Learning Shapes},
     year = {2020}
    }
    	
## Related papers
* [Yariv et al. - Multiview Neural Surface Reconstruction with Implicit Lighting and Material](https://arxiv.org/abs/2003.09852)
* [Atzmon & Lipman. - SAL++: Sign Agnostic Learning with Derivatives (2020)](https://arxiv.org/abs/2006.05400)
* [Atzmon & Lipman. - SAL: Sign Agnostic Learning of Shapes From Raw Data (2020)](https://arxiv.org/abs/1911.10414)
* [Atzmon et al. - Controlling Neural Level Sets (2019)](https://arxiv.org/abs/1905.11911)
	
