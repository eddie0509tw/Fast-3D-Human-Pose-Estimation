# Fast-3D-Human-Pose Estimation

### Introduction
This is a pytorch implementation of method based on [Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation](https://arxiv.org/pdf/2004.02186.pdf) applying on stereo human pose estimation tasks. We also compare this with simply implement a naive approach reference to [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/abs/1804.06208) which consist of encoder decoder structure to get a 2d pose from both view.  We compare their Mean Per Joint Position Error (MPJPE) as metric for both 2D and 3D case, and we also using some data augmentation tricks like masking out a small block on human for one of the image, for example Cutout [https://arxiv.org/abs/1708.04552], Hide-and-Seek [https://arxiv.org/pdf/1704.04232.pdf] to improve accuracy.

### Contribution
- We implement and extent the method [Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation](https://arxiv.org/pdf/2004.02186.pdf) from scratch with slightly modification and apply it to the stereo reconstruction tasks which first train on human pose dataset (MPII) and fine-tune on the stereo images dataset (MADS).
-  We find that the random masking data augmentation strategies can ease the self-occlusion and improve the MPJPE to some extent.
- We experiment with different tricks (Different Loss Function, Gradient Clip) to further increase both the accuracy and stabilization for the training process.

#### Dataset

We pretrained our model using the [MPII](http://human-pose.mpi-inf.mpg.de/) Dataset which includes around 25K images containing over 40K people with annotated body joints. The images were systematically collected using an established taxonomy of every day human activities. Then we do fine-tuning on the stereo data from [MADS](http://visal.cs.cityu.edu.hk/research/mads/#download) Dataset which consists of martial arts actions (Tai-chi and Karate), dancing actions (hip-hop and jazz), and sports actions (basketball, volleyball, football, rugby, tennis and badminton). Two martial art masters, two dancers and an athlete performed these
actions while being recorded with either multiple cameras or a stereo depth camera.

### Train

Instuction: Todo

### Test
 
Instuction: Todo

### Results 

Best:

<img src="https://github.com/eddie0509tw/Fast-3D-Human-Pose-Estimation/blob/main/GIF/HipHop_best.gif" alt="HipHop_best" />
<img src="https://github.com/eddie0509tw/Fast-3D-Human-Pose-Estimation/blob/main/GIF/Sports_best.gif" alt="Sports_best" />
Baseline:

<img src="https://github.com/eddie0509tw/Fast-3D-Human-Pose-Estimation/blob/main/GIF/HipHop_baseline.gif" alt="HipHop_base" />
<img src="https://github.com/eddie0509tw/Fast-3D-Human-Pose-Estimation/blob/main/GIF/Sports_baseline.gif" alt="Sports_base" />

### References

[CDRNet](https://github.com/TemugeB/CDRnet/tree/main)</br>
[DiffDLT](https://github.com/edoRemelli/DiffDLT/blob/master/dlt.py)</br>
[learnable-triangulation-pytorch](https://github.com/karfly/learnable-triangulation-pytorch)</br>
[R-YOLOv4](https://github.com/kunnnnethan/R-YOLOv4/tree/main)</br>

