# DeepUnet_Keras_SE
Keras implementation of Deep Unet with SE block

DeepUnet paper:
- https://arxiv.org/abs/1709.00201


Basic structure of Unet in this implementation is based on
- http://ni4muraano.hatenablog.com/entry/2017/08/10/101053

Requirement:
- tensorflow-gpu > 1.13.2, but not tensorflow-gpu > 2.0
  (for 1.13.2 CUDA = 10.4, CuDNN = 7.4)

Usage:
- First, download dataset:
 [Daimler Pedestrian Segmaneation](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Daimler_Pedestrian_Segmentatio/daimler_pedestrian_segmentatio.html)
- $ mkdir check_points results

Edditing some paramters in train_and_prediction_aug.py.
For instance, number of epochs (more than 200 epochs is recommended for deepunet + SE model).

- $ python train_and_prediction_aug.py


***
Results:

(Left) Simple Unet  (Right)  Deep Unet
![Results](https://github.com/TKouyama/DeepUnet_Keras/blob/master/images/Unet_deep_rev01.png)

***

Memo:

- Data augmentaiton (horizontal flip) significantly improved the segmentation results.
- Deep Unet seems to improve detail of segmentation area.
- Merged Loss of dice loss and binary cross entropy provides better segmentation shape.
- SE block may improve the detal shape a little ?

