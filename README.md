# Dual-Stream Fusion Network for Spatiotemporal Video Super-Resolution (WACV 2021)

Min-Yuan Tseng, [Yen-Chung Chen](https://yenchungchen.github.io/), Yi-Lun Lee, [Wei-Sheng Lai](https://www.wslai.net/), [Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/), [Wei-Chen Chiu](https://walonchiu.github.io/)

[Winter Conference on Applications of Computer Vision (WACV), 2021](http://wacv2021.thecvf.com/home)

[[Paper]](http://people.cs.nctu.edu.tw/~walon/publications/tseng2021wacv.pdf)
[[Video]](http://people.cs.nctu.edu.tw/~walon/publications/tseng2021wacv_video.mp4)
[[Supplementary]](http://people.cs.nctu.edu.tw/~walon/publications/tseng2021wacv_supp.pdf)

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
You can download processed vimeo90k lmdb files from this [link](https://drive.google.com/drive/folders/11_g_iSkLFntqKwtccgZPAAoB5RwgZezh?usp=sharing).
If you want to process the original [vimeo90k](http://toflow.csail.mit.edu/) dataset on your own, you can use `dataset/create_lmdb_mp.py`.

## Evaluation
Download the pretrained weights (`ESPCN.pth`, `SuperSloMo.pth` and `STSR_best.pth`) from this [link](https://drive.google.com/drive/folders/1JLAc1wtfPzvxUfp8T0irEKf1N8GbhCo9?usp=sharing) and place them in `pretrained`, then run:

```
python test_model.py --data_root ./data/vimeo90k/ --sr_type ESPCN --it_type SSM --two_mask --forward_MsMt --forward_F --forward_R --stsr_weight ./pretrained/STSR_best.pth --sr_weight ./pretrained/ESPCN.pth --it_weight ./pretrained/SuperSloMo.pth --batch_size 24
```

## Train
run:

```
python train.py --data_root ./data/vimeo90k/ --sr_type ESPCN --it_type SSM --two_mask --forward_MsMt --forward_F --forward_R --batch_size 24
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

## Acknowledgments
Based on different architectures, we modify the source codes from [SuperSloMo](https://github.com/avinashpaliwal/Super-SloMo), [DAIN](https://github.com/baowenbo/DAIN) and [SAN](https://github.com/daitao/SAN). We also use the script for processing images to lmdb from [EDVR](https://github.com/xinntao/EDVR).
