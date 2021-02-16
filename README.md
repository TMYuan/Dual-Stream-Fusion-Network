# Dual-Stream Fusion Network for Spatiotemporal Video Super-Resolution (WACV 2021)

Min-Yuan Tseng, [Yen-Chung Chen](https://yenchungchen.github.io/), Yi-Lun Lee, [Wei-Sheng Lai](https://www.wslai.net/), [Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/), [Wei-Chen Chiu](https://walonchiu.github.io/)

[Winter Conference on Applications of Computer Vision (WACV), 2021](http://wacv2021.thecvf.com/home)

[[Paper]](http://people.cs.nctu.edu.tw/~walon/publications/tseng2021wacv.pdf)[[Video]](http://people.cs.nctu.edu.tw/~walon/publications/tseng2021wacv_video.mp4)[[Supplementary]](http://people.cs.nctu.edu.tw/~walon/publications/tseng2021wacv_video.mp4)

## Environment
Python3.6

PyTorch 1.1.0

CUDA 9.0

Other dependencies are listed in `environment.yml`
Use conda to create the environment and activate it:
```
conda env create -f environment.yml
conda activate dual-stream
```

## Prepare dataset
We use `lmdb` as dataset to reduce time for loading image.
You can download processed vimeo90k lmdb files from this [link]().
If you want to process the dataset on your own, you can use `dataset/create_lmdb_mp.py`

## Evaluation
Download the pretrained weights from this [link]() and place them in `pretrained`, then run:

```
python test_model.py --data_root ./data/vimeo90k/ --sr_type ESPCN --it_type SSM --two_mask --forward_MsMt --forward_F --forward_R --stsr_weight ./pretrained/STSR_best.pth --sr_weight ./pretrained/ESPCN_33.pth --it_weight ./pretrained/SuperSloMo_132.pth --batch_size 24
```

## Citation
If you find this work useful for your research, please cite:
```
@inproceedings{tseng21wacv,
 title = {Dual-Stream Fusion Network for Spatiotemporal Video Super-Resolution},
 author = {Min-Yuan Tseng and Yen-Chung Chen and Yi-Lun Lee and Wei-Sheng Lai and Yi-Hsuan Tsai and Wei-Chen Chiu},
 booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
 year = {2021}
}
```
