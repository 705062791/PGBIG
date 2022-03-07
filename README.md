# Progressively-Generating-Better-Initial-Guesses-Towards-Next-Stages-forHigh-Quality-Human-Motion-Pre

Official implementation of [Progressively Generating Better Initial Guesses Towards Next Stages for High-Quality
Human Motion Prediction](assets/07627.pdf) (CVPR 2022 paper)

PDF Sup Demo (Coming soon) 

[comment]: <> ([PDF]&#40;&#41; [Supp]&#40;&#41; [Demo]&#40;&#41;)

[comment]: <> ([\[PDF\]]&#40;assets/07627.pdf&#41;  [\[Supp\]]&#40;assets/07627-supp.pdf&#41;)

## Authors

1. [Tiezheng Ma](https://github.com/705062791), School of Computer Science and Engineering, South China University of Technology, China, [mtz705062791@gmail.com](mailto:mtz705062791@gmail.com)
2. [Yongwei Nie](https://nieyongwei.net), School of Computer Science and Engineering, South China University of Technology, China, [nieyongwei@scut.edu.cn](mailto:nieyongwei@scut.edu.cn)
3. [Chengjiang Long](http://www.chengjianglong.com), JD Finance America Corporation, USA, [cjfykx@gmail.com](mailto:cjfykx@gmail.com)
4. [Qing Zhang](http://zhangqing-home.net/), School of Computer Science and Engineering, Sun Yat-sen University, China, [zhangqing.whu.cs@gmail.com](mailto:zhangqing.whu.cs@gmail.com)
5. [Guiqing Li](http://www2.scut.edu.cn/cs/2017/0629/c22284a328097/page.htm), School of Computer Science and Engineering, South China University of Technology, China, [ligq@scut.edu.cn](mailto:ligq@scut.edu.cn)

## Abstract
######  &nbsp;&nbsp;&nbsp;  This paper presents a high-quality human motion prediction method that accurately predicts future human poses given observed ones. Our method is mainly based on the observation that a good initial guess of the future pose sequence, such as the mean of future poses, is very helpful to improve the forecasting accuracy. This motivates us to design a novel two-stage prediction strategy, including an init-prediction network that just computes a good initial guess and a formal-prediction network that takes both the historical and initial poses to predict the target pose sequence. We extend this idea further and design a multi-stage prediction framework with each stage predicting initial guess for the next stage, which rewards us with significant performance gain. To fulfill the prediction task at each stage, we propose a network comprising Spatial Dense Graph Convolutional Networks (S-DGCN) and Temporal Dense Graph Convolutional Networks (T-DGCN). Sequentially executing the two networks can extract spatiotemporal features over the global receptive field of the whole pose sequence effectively. All the above design choices cooperating together make our method outperform previous approaches by a large margin (6\%-7\% on Human3.6M, 5\%-10\% on CMU-MoCap, 13\%-16\% on 3DPW).

## Dependencies

* Pytorch 1.8.0+cu11
* Python 3.7
* Nvidia RTX 2060

## DataSet
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

[CMU mocap](http://mocap.cs.cmu.edu/) was obtained from the [repo](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) of ConvSeq2Seq paper.

[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website.

## Train
+ Train on Human3.6M:

`
python main_h36m.py
  --data_dir
[dataset path]
--kernel_size
10
--dct_n
35
--input_n
10
--output_n
25
--skip_rate
1
--batch_size
16
--test_batch_size
32
--in_features
66
--cuda_idx
cuda:0
--d_model
16
--lr_now
0.005
--epoch
50
--test_sample_num
-1
  `

+ Train on CMU-MoCap:

`
python main_cmu_3d.py
--data_dir
[dataset path]
--kernel_size
10
--dct_n
35
--input_n
10
--output_n
25
--skip_rate
1
--batch_size
16
--test_batch_size
32
--in_features
75
--cuda_idx
cuda:0
--d_model
16
--lr_now
0.005
--epoch
50
--test_sample_num
-1
`

Train on 3DPW:

`
--data_dir
[dataset path]
--kernel_size
10
--dct_n
40
--input_n
10
--output_n
30
--skip_rate
1
--batch_size
32
--test_batch_size
32
--in_features
69
--cuda_idx
cuda:0
--d_model
16
--lr_now
0.005
--epoch
50
--test_sample_num
-1
`

**Note**: 
+ `kernel_size`: is the length of used input seqence.
  
+ `d_model`: is the latent code dimension of a joint.
  
+ `test_sample_num`: is the sample number for test dataset, can be set as `{8, 256, -1(all)}`. For example, if it is set to `8`, it means that 8 samples are sampled for each action as the test set.

After training, the checkpoint is saved in `./checkpoint/`.
## Test
Add `--is_eval` after the above training commands. 

The test result will be saved in `./checkpoint/`.

## Citation

If you think our work is helpful to you, please cite our paper.

```
Coming soon

```

[comment]: <> (```)

[comment]: <> (@inproceedings{lingwei2021msrgcn,)

[comment]: <> (  title={MSR-GCN: Multi-Scale Residual Graph Convolution Networks for Human Motion Prediction},)

[comment]: <> (  author={Lingwei, Dang and Yongwei, Nie and Chengjiang, Long and Qing, Zhang and Guiqing Li},)

[comment]: <> (  booktitle={Proceedings of the IEEE International Conference on Computer Vision &#40;ICCV&#41;},)

[comment]: <> (  year={2021})

[comment]: <> (})

[comment]: <> (```)

## Acknowledgments
Our code is based on [HisRep](https://github.com/wei-mao-2019/HisRepItself) and [LearnTrajDep](https://github.com/wei-mao-2019/LearnTrajDep)

[comment]: <> (Some of our evaluation code and data process code was adapted/ported from [LearnTrajDep]&#40;https://github.com/wei-mao-2019/LearnTrajDep&#41; by [Wei Mao]&#40;https://github.com/wei-mao-2019&#41;. )

## Licence
MIT