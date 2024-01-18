# Fast-3D-Human-Pose Estimation

### Introduction
This is a pytorch implementation of method based on [Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation](https://arxiv.org/pdf/2004.02186.pdf) applying on stereo images to reconstruct the human poses in 3D world. We also compare this with a naive approach reference to [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/abs/1804.06208) which consist of encoder decoder structure and predict 2d pose from both view.  We evaluate their performance using the Mean Per Joint Position Error (MPJPE) metric in both 2D and 3D scenarios. Additionally, we employ data augmentation techniques, such as masking out a small block on the human in images, incorporating methods like [Cutout](https://arxiv.org/abs/1708.04552) and [Hide-and-Seek](https://arxiv.org/pdf/1704.04232.pdf), to enhance the accuracy of the models.

### Contribution
- We implement and extent the method [Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation](https://arxiv.org/pdf/2004.02186.pdf) from scratch with slightly modification and apply it to the stereo reconstruction tasks.
-  We find that the random masking data augmentation strategies can more or less ease the self-occlusion and improve the MPJPE to some extent.
- We experiment with different tricks (Different Loss Function, Gradient Clip) to further increase both the accuracy and stabilization for the training process.

### Dataset

We pretrained our model using the [MPII](http://human-pose.mpi-inf.mpg.de/) Dataset which includes around 25K images containing over 40K people with annotated body joints. Then we do fine-tuning on the stereo data from [MADS](http://visal.cs.cityu.edu.hk/research/mads/#download) Dataset which consists of martial arts actions (Tai-chi and Karate), dancing actions (hip-hop and jazz), and sports actions (basketball, volleyball, football, rugby, tennis and badminton). Two martial art masters, two dancers and an athlete performed these
actions while being recorded with either multiple cameras or a stereo depth camera.

### Train

Instuction: Todo

### Test
 
Instuction: Todo

### Weights
You can download the best weight via the [link](https://drive.google.com/drive/folders/1wEZc3rrNR-Erb9fofr2fEvK0ba27t5y-?usp=sharing)

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

