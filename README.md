# DeepUnet_Keras
Keras implementation of Deep Unet

DeepUnet paper:
- https://arxiv.org/abs/1709.00201


Basic structure of Unet in this implementation is based on
- http://ni4muraano.hatenablog.com/entry/2017/08/10/101053

Usage:
- First, download dataset:
 [Daimler Pedestrian Segmaneation](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Daimler_Pedestrian_Segmentatio/daimler_pedestrian_segmentatio.html)
- mkdir check_points results
- python train_and_prediction_aug.py


***
Results:

(Left) Simple Unet  (Right)  Deep Unet
![Results](https://github.com/TKouyama/DeepUnet_Keras/blob/master/images/Unet_deep_rev01.png)

***

Memo:

- Data augmentaiton (horizontal flip) significantly improved the segmentation results.
- Deep Unet seems to improve detail of segmentation area.
- Merged Loss with dice loss and bce provides better segmentation shape.

