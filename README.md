# Fast-3D-Human-Pose Estimation

### Introduction
In this project, we’re going to explore the methods on dealing with self-occlusion occurs in 3d human pose estimation model(2-views) efficiently. We began with simply implement a naive approach [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/abs/1804.06208) which consist of encoder decoder structure to get a 2d pose from both view, and we then could triangulate the corresponding points to get 3d pose. We then compare this
naive approach with this method based on [Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation](https://arxiv.org/pdf/2004.02186.pdf). We’ll further investigate output results in those methods to see under occlusion what will it predict, comparing their Mean Per Joint Position Error (MPJPE), we might also explore using some data augmentation tricks like masking out a small block on human for one of the image, for example Cutout [https://arxiv.org/abs/1708.04552], Hide-and-Seek [https://arxiv.org/pdf/1704.04232.pdf] to provide generalization of occlusion problems.

### Contribution
(1) We implement and extent the method [Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation](https://arxiv.org/pdf/2004.02186.pdf) from scratch to the stereo reconstruction tasks which first train on human pose dataset (MPII) and fine-tune on the stereo images dataset (MADS).
(2) We show that learning the disentangled representation using FTL and fusion layer can improve the self-occlusion
as well as the Mean Per Joint Position Error (MPJPE).
(3) We show that our random masking data augmentation strategies can ease the self-occlusion to some extent.
(4) We experiment with different tricks (Different Loss Function, Gradient Clip) to further increase both the accuracy and stabilization for the training process
### Train

Instuction: Todo

#### Dataset

We pretrained our model using the MPII[http://human-pose.mpi-inf.mpg.de/] Dataset which includes around 25K images containing over 40K people with annotated body joints. The images were systematically collected using an established taxonomy of every day human activities. Then we do fine-tuning on the stereo data from MADS[http://visal.cs.cityu.edu.hk/research/mads/#download] Dataset which consists of martial arts actions (Tai-chi and Karate), dancing actions (hip-hop and jazz), and sports actions (basketball, volleyball, football, rugby, tennis and badminton). Two martial art masters, two dancers and an athlete performed these
actions while being recorded with either multiple cameras or a stereo depth camera.



### Test
 
Instuction: Todo

### Results 
### References

[CDRNet](https://github.com/TemugeB/CDRnet/tree/main)</br>
[DiffDLT](https://github.com/edoRemelli/DiffDLT/blob/master/dlt.py)</br>
[learnable-triangulation-pytorch](https://github.com/karfly/learnable-triangulation-pytorch)</br>
[R-YOLOv4](https://github.com/kunnnnethan/R-YOLOv4/tree/main)</br>

