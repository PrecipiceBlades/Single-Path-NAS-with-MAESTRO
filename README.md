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
| MnasNet-A1           | 3887038(3.9M)| 330M      | 25.3M  | 11.5s             |
| MobileNet-V2         | 3504872(3.5M)| 320M      | 24.0M  | 10.9s             |
| MobileNet-V3(large)  | 5476416(5.5M)| 233M      | 19.8M  | 9.0s              |
| MobileNet-V3(small)  | 2534656(2.5M)| 65M       | 6.39M  | 2.9s              |
| ProxylessNet(mobile) | 4080512(4.1M)| 336M      | 26.2M  | 11.9s             |
| SinglepathNAS        | 4414216(4.4M)| 360M      | 29.8M  | 13.5s             |
