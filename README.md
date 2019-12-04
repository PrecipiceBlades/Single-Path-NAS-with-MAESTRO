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
| MnasNet-A1           | 3.94M(3.9M)  | 312M      | 19.2M  | 8.72s             |
| MobileNet-V2         | 3.50M(3.5M)  | 300M      | 24.0M  | 11s               |
| MobileNet-V3(small)  | 5.43M(5.4M)  | 219M      | 6.04M  | 2.72s             |
| MobileNet-V3(large)  | 2.47M(2.5M)  | 56M       | 19.5M  | 8.73s             |
| ProxylessNet(mobile) | 4.08M(4.08M) | 340M      | 26.2M  | 11.9s             |
