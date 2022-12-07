# GeoDSR
## Learning Continuous Depth Representation via Geometric Spatial Aggregator
## Accepted by AAAI 2023 
Xiaohang Wang*, Xuanhong Chen*, Bingbing Ni**, Zhengyan Tong, Hang Wang

\* Equal contribution

\*\* Corresponding author



**The official repository with Pytorch**

[![geodsrlogo](/docs/img/geodsrlogo.png)](https://github.com/nana01219/GeoDSR)


## Dependencies
- python3.7+
- pytorch1.9+
- torchvision
- [Nvidia Apex](https://github.com/NVIDIA/apex) (python-only build is ok.)


### Dataset
We follow [Tang et al.](https://github.com/ashawkey/jiif) and use the same datasets. Please refer to [here](https://github.com/ashawkey/jiif/blob/main/data/prepare_data.md) to download the preprocessed dataset and extract them into `data/` folder.

### Pretrained Models
- Baidu Netdisk (百度网盘)：https://pan.baidu.com/s/1e2rLQFqVHIy2ZZG922XNTA 
- Extraction Code (提取码)：xu7e

Please put the model under `workspace/checkpoints` folder.

### Train
```
python main.py
```
### Test
```
bash test.sh
```




## Licesnse
For academic and non-commercial use only. The whole project is under the MIT license. See [LICENSE](https://github.com/nana01219/GeoDSR/blob/main/LICENSE) for additional details.


## Citation
If you find this project useful in your research, please consider citing:

```

```

## Ackownledgements
This code is built based on [JIIF](https://github.com/ashawkey/jiif). We thank the authors for sharing the codes.

## Related Projects

Learn about our other projects 

[[VGGFace2-HQ]](https://github.com/NNNNAI/VGGFace2-HQ): high resolution face dataset VGGFace2-HQ;

[[RainNet]](https://neuralchen.github.io/RainNet);

[[Sketch Generation]](https://github.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale);

[[CooGAN]](https://github.com/neuralchen/CooGAN);

[[Knowledge Style Transfer]](https://github.com/AceSix/Knowledge_Transfer);

[[SimSwap]](https://github.com/neuralchen/SimSwap): popular face swapping project;

[[ASMA-GAN]](https://github.com/neuralchen/ASMAGAN): high-quality style transfer project;
