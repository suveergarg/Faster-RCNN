# RCNN

In this project I implemented MaskRCNN, an algorithm that addresses the task of instance seg-
mentation, which combines object detection and semantic segmentation into a per-pixel object detection framework. We train the Network on the standard COCO dataset, which has 80 object classes.

### Network Architecture
![](images/NetworkArchitecture.png)
![](images/network1.png)
![](images/network2.png)

### Loss Objective
![](images/Loss1.png)

### Ground Truth and Outputs From Region Proposal Network
Region Proposal Networks (RPNs) are "attention mechanisms" for the object detection task, performing a crude but inexpensive first estimation of where the bounding boxes of the objects should be. 

![](images/GTandRPN.png)
![](images/GTandRPN2.png)

### Output From Network Before Post-Processing
![](images/networkOutput.png)
![](images/networkOutput1.png)


### After Post-Processing
![](images/postProc.png)
![](images/postProc1.png)

### Loss Plots
![](images/Losses.png)

### MAP
![](images/MAP.png)



