# VOC Example


## Training


```bash
./download_dataset.sh  # 下载并解压VOC的数据

./train_fcn32s.py -g 0  # 开始训练fcn32s
``` 
VGG16 pth参数地址：
/media/atr/新加卷/WMJ/model_pth/vgg16_from_caffe.pth

VOC 数据存放地址：
/media/atr/Seagate Expansion Drive/dataset/VOC

- http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
==>>/media/atr/Seagate Expansion Drive/dataset/VOC/benchmark_RELEASE
- http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
==>>/media/atr/Seagate Expansion Drive/dataset/VOC/VOCdevkit