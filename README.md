# single-path-nas-pytorch
Pytorch version of codebase for the paper single-path-nas

Use Test.py to test Pytorch & Tensorflow equivalance on Depthwise Convolution 

# Experiments

Configuration:

L1 cache: 98

L2 Cache: 5408

Frequency: 2.2G Hz

| Model                | Params       | Multi&Add | Cycles | Estimated Runtime |
|----------------------|--------------|-----------|--------|-------------------|
| MnasNet-A1           | 3887038(3.9M)| 330M      | 19.2M  | 8.72s             |
| MobileNet-V2         | 3504872(3.5M)| 320M      | 24.0M  | 11s               |
| MobileNet-V3(large)  | 5476416(5.5M)| 233M      | 6.04M  | 2.72s             |
| MobileNet-V3(small)  | 2534656(2.5M)| 65M       | 19.5M  | 8.73s             |
| ProxylessNet(mobile) | 4080512(4.1M)| 336M      | 26.2M  | 11.9s             |
| SinglepathNAS        | 4414216(4.4M)| 360M      | 44.1M  | 11.9s             |
