# FOS Test Datasets

## Data Source
Here we introduce how the FOS test datasets are obtained.

### Real-world data
Real-world test datasets includes FOS-V and FOS-real subsets.

#### FOS-V
<img src='figures/fos_v_samples.gif' width="100%" align='center' title="Samples from FOS-V">

<br>
All real-world data of FOS-V test datasets are derived from the following two parts. 
- Videos from the publicly available datasets, [YTCeleb](http://seqamlab.com/youtube-celebrities-face-tracking-and-recognition-dataset/") and [YTFace](https://www.cs.tau.ac.il/\~wolf/ytfaces/).
- Self-collected video data from YouTube, named as YTW, for which we provide the source metadata in forms of YouTube video id as <a href="https://pan.baidu.com/s/1-ZgAb1Ianm0xbu-oYpP8Zg?pwd=98m8" target="_blank">YTW meta</a>.


All real data from FOS-V test datasets are named following a naming rule of **{dataset id}_{data id}[_data frame id].{suffix}**. The dataset ID mapping is listed as follows.

| Dataset | Dataset ID |
|:---:|:---:|
|[YTFace](https://www.cs.tau.ac.il/\~wolf/ytfaces/)| 1|
|[YTCeleb](http://seqamlab.com/youtube-celebrities-face-tracking-and-recognition-dataset/)| 2|
|YTW(self-collected)| 3|


With all raw videos collected, they were processed by face tracking and cropping to form FOS-V. The processing scripts and config files can be found in this [module](https://github.com/ziyannchen/VFRxBenchmark/tree/main/bfrxlib/preprocess). 
We provide the processing metadata of each clip to obtain FOS-V from raw video collections as <a href="https://pan.baidu.com/s/1mxITm8zEr7JGmW77pWQpIA?pwd=shha">FOS-V meta</a>.

#### FOS-real
<img src='figures/fos_real_samples.png' width="100%" align='center' title="Samples from FOS-real">

<br>

All images of FOS-real are from frames of FOS-V. The image name has the same naming rule of frames in FOS-V, i.e. **{dataset id}\_{data id}\_{data frame id}.{suffix}**.

### Synthetic data


Synthetic dataset includes only the FOS-syn subset.

#### FOS-syn
<img src='figures/fos_syn_samples.png' width="100%" title="Samples from FOS-syn" align='center'>

FOS-syn test set was synthesized based on a subset of the <a href="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html">CelebA-HQ</a> Test(5k) dataset.
The degradation processing script can be found in [here](https://github.com/ziyannchen/VFRxBenchmark/blob/main/scripts/make_synthetic_deg_data.py) with reference to [GFPGAN](https://github.com/TencentARC/GFPGAN).
