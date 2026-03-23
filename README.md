<div align="center">
<h1>LitePT: Lighter Yet Stronger Point Transformer</h1>

[**Yuanwen Yue**](https://ywyue.github.io/), [**Damien Robert**](https://drprojects.github.io/), [**Jianyuan Wang**](https://jytime.github.io/),
[**Sunghwan Hong**](https://sunghwanhong.github.io/), [**Jan Dirk Wegner**](https://dm3l.uzh.ch/wegner/group-leader)<br>[**Christian Rupprecht**](https://chrirupp.github.io/),
[**Konrad Schindler**](https://igp.ethz.ch/personen/person-detail.html?persid=143986)

ETH Zurich,
University of Oxford,
University of Zurich

**CVPR 2026**

<a href="https://arxiv.org/abs/2512.13689"><img src="https://img.shields.io/badge/arXiv-LitePT-red" alt="Paper PDF"></a>
<a href="https://litept.github.io/"><img src="https://img.shields.io/badge/Project_Page-LitePT-green" alt="Project Page"></a>
<a href='https://huggingface.co/prs-eth/LitePT'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt="HF Model"></a>

<img
  src="https://raw.githubusercontent.com/LitePT/LitePT.github.io/main/assets/figures/teaser.png"
  alt="teaser"
  width="800"
/>

</div>

**LitePT** is a lightweight, high-performance 3D point cloud architecture.
<u>Left</u>: LitePT-S has $3.6\times$ fewer parameters, $2\times$ faster runtime and $2\times$ lower memory footprint than PTv3. Moreover, it remains faster and more memory-efficient even when scaled up to LitePT-L with a parameter count twice that of PTv3. <u>Right</u>: Already the smallest variant, LitePT-S, matches or outperforms state-of-the-art point cloud backbones across a range of benchmarks.

## News
- **2026-02-21:** LitePT is accepted to CVPR 2026.
- **2025-12-16:** Paper, project page, code, models are released.

## Preparation

### Environment
- Create an environment and install pytorch and other required packages:
  ```shell
  git clone https://github.com/prs-eth/LitePT.git
  cd LitePT
  conda create -n litept python=3.10
  conda activate litept
  # install pytorch, adjusting the command to match your cuda version
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
  # install some other packages
  pip install -r requirements.txt
  # spconv (SparseUNet)
  pip install spconv-cu124
  # flash attention
  pip install git+https://github.com/Dao-AILab/flash-attention.git
  ```
- PointROPE. Modify the ```all_cuda_archs``` in ```libs/pointrope/setup.py``` to your GPU arch, e.g. 8.6: GeForce RTX 3090; 9.0: NVIDIA H100; more info: https://developer.nvidia.com/cuda/gpus
  ```shell
  cd libs/pointrope
  python setup.py install
  cd ../..
  ```
  To support users who may want to use PointROPE in non-CUDA environments, we also provide a pure PyTorch implementation of PointROPE. The CUDA-compiled version is used by default. If it is unavailable, the system will automatically fall back to the PyTorch implementation. Please note that the PyTorch version is slightly slower than the CUDA-optimized implementation. Benchmark results are shown below:
  <details style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Benchmark: PointROPE (Pure PyTorch VS. CUDA)</summary>

  <table>
   <caption style="caption-side: bottom; text-align: center">Efficiency comparison on ScanNet.</caption>
   <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2"><div align="center">Training</div></th>
      <th colspan="2"><div align="center">Inference</div></th>
    </tr>
    <tr>
      <th>Method</th>
      <th>#Params</th>
      <th>Latency</th>
      <th>Memory</th>
      <th>Latency</th>
      <th>Memory</th>
    </tr>
   </thead>
   <tbody>
    <tr>
      <td>PTv3</td>
      <td>46.1M</td>
      <td>110ms</td>
      <td>5.8G</td>
      <td>51ms</td>
      <td>4.1G</td>
    </tr>
    <tr>
      <td>LitePT-S (Pure Torch PointROPE)</td>
      <td>12.7M</td>
      <td>78ms</td>
      <td>2.4G</td>
      <td>24ms</td>
      <td>2.0G</td>
    </tr>
    <tr>
    <td>LitePT-S (CUDA PointROPE)</td>
      <td>12.7M</td>
      <td>72ms</td>
      <td>2.3G</td>
      <td>21ms</td>
      <td>2.0G</td>
    </tr>
   </tbody>
  </table>

  </details>

- Optional: Install CUDA kernels for faster point cloud serialization in attention layers (automatic fallback to the original PyTorch implementation). Modify the ```all_cuda_archs``` in ```libs/serialization/setup.py``` (see PointROPE above).
  ```shell
  cd libs/serialization
  python setup.py install
  cd ../..
  ```

- Additional requirements. The requirements below are optional, and only required for evaluator and PointGroup instance segmentation. 
  * For evaluator:
    ```
    cd libs/pointops
    python setup.py install
    cd ../..
    ```
  * For PointGroup:
    ```
    conda install -c bioconda google-sparsehash 
    cd libs/pointgroup_ops
    python setup.py install
    cd ../..
    ```

### Use LitePT in Your Own Project
To simplify integrating LitePT into your own projects, we provide a standalone implementation of LitePT. To include LitePT in your codebase, copy the directories ```LitePT/litept``` and ```LitePT/libs/pointrope``` into your project, arranging them as shown below:

```
Your_project/
└── .../
    └── LitePT/
        ├── litept/
        ├── libs/serialization
        └── libs/pointrope
```
We also provide a ```demo_use.py``` script that illustrates how to use the standalone LitePT with an example point cloud input.

### Data
We prepare and organize the data in the same way with Pointcept [Data Preparation](https://github.com/Pointcept/Pointcept#data-preparation). All data should be placed in ```LitePT/data```.


## Model Zoo

### Semantic segmentation 
| Model | Params | Benchmark  | val mIoU | Config | Checkpoint |
|:-|-:|:-:|:-:|:-:|:-:|
| LitePT-S | 12.7M | NuScenes | 82.2 | [link](https://github.com/prs-eth/LitePT/blob/main/configs/nuscenes/semseg-litept-small-v1m1.py) | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/nuscenes-semseg-litept-small-v1m1/model/model_best.pth) |
| LitePT-S | 12.7M | Waymo | 73.1 |[link](https://github.com/prs-eth/LitePT/blob/main/configs/waymo/semseg-litept-small-v1m1.py) | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/waymo-semseg-litept-small-v1m1/model/model_best.pth) |
| LitePT-S | 12.7M | ScanNet  | 76.5 |[link](https://github.com/prs-eth/LitePT/blob/main/configs/scannet/semseg-litept-small-v1m1.py) | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/scannet-semseg-litept-small-v1m1/model/model_best.pth) |
| LitePT-S | 12.7M | Structured3D | 83.6 | [link](https://github.com/prs-eth/LitePT/blob/main/configs/structured3d/semseg-litept-small-v1m1.py) | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/structured3d-semseg-litept-small-v1m1/model/model_best.pth) |
| LitePT-B | 45.1M | Structured3D | 85.1  | [link](https://github.com/prs-eth/LitePT/blob/main/configs/structured3d/semseg-litept-base-v1m1.py) | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/structured3d-semseg-litept-base-v1m1/model/model_best.pth) |
| LitePT-L | 85.9M | Structured3D | 85.4 | [link](https://github.com/prs-eth/LitePT/blob/main/configs/structured3d/semseg-litept-large-v1m1.py) | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/structured3d-semseg-litept-large-v1m1/model/model_best.pth) |

### Instance segmentation 
| Model | Params | Benchmark  | mAP<sub>25</sub> | mAP<sub>50</sub> | mAP | Config | Checkpoint |
|:-|-:|:-:|:-:|:-:|:-:|:-:|:-:|
| LitePT-S* | 16.0M | ScanNet | 78.5 | 64.9 | 41.7 | [link](https://github.com/prs-eth/LitePT/blob/main/configs/scannet/insseg-litept-small-v1m2.py) | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/scannet-insseg-litept-small-v1m2/model/model_best.pth) |
| LitePT-S* | 16.0M | ScanNet200 | 40.3 | 33.1 | 22.2 | [link](https://github.com/prs-eth/LitePT/blob/main/configs/scannet200/insseg-litept-small-v1m2.py) | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/scannet200-insseg-litept-small-v1m2/model/model_best.pth) |
### Object detection
| Model | Params | Benchmark  | mAPH | Config | Checkpoint |
|:-|-:|:-:|:-:|:-:|:-:|
| LitePT | 9.0M | Waymo  | 70.7 | link | [Download](https://huggingface.co/prs-eth/LitePT/blob/main/waymo-objdet-litept-small-v1m3/model/model_best.pth) |


## Training

###  Semantic segmentation
```shell
### NuScenes + LitePT-S
sh scripts/train.sh -g 4 -d nuscenes -c semseg-litept-small-v1m1 -n semseg-litept-small-v1m1
### Waymo + LitePT-S
sh scripts/train.sh -g 4 -d waymo -c semseg-litept-small-v1m1 -n semseg-litept-small-v1m1
### ScanNet + LitePT-S
sh scripts/train.sh -g 4 -d scannet -c semseg-litept-small-v1m1 -n semseg-litept-small-v1m1
### Structured3D + LitePT-S
sh scripts/train.sh -g 16 -d structured3d -c semseg-litept-small-v1m1 -n semseg-litept-small-v1m1
### Structured3D + LitePT-B
sh scripts/train.sh -g 16 -d structured3d -c semseg-litept-base-v1m1 -n semseg-litept-base-v1m1
### Structured3D + LitePT-L
sh scripts/train.sh -g 16 -d structured3d -c semseg-litept-large-v1m1 -n semseg-litept-large-v1m1
```

###  Instance segmentation
```shell
### ScanNet + LitePT-S*
sh scripts/train.sh -g 4 -d scannet -c insseg-litept-small-v1m2 -n insseg-litept-small-v1m2

### ScanNet200 + LitePT-S*
sh scripts/train.sh -g 4 -d scannet200 -c insseg-litept-small-v1m2 -n insseg-litept-small-v1m2
```

## Testing

###  Semantic segmentation
The training pipeline automatically runs testing once training is complete through the ```PreciseEvaluator``` hook. Alternatively, standalone testing with pretrained weights can be run as follows:

```shell
export PYTHONPATH=./
python tools/test.py --config-file "${CONFIG_PATH}" --num-gpus "${NUM_GPU}" --options save_path="${SAVE_PATH}" weight="${CHECKPOINT_PATH}"

## E.g. for NuScenes + LitePT-S
## python tools/test.py --config-file configs/nuscenes/semseg-litept-small-v1m1.py --num-gpus 4 --options save_path=exp/nuscenes/semseg-litept-small-v1m1 weight=exp/nuscenes/semseg-litept-small-v1m1/model/model_best.pth
```



## Checklist
- [x] Release models, code for semantic segmentation and instance segmentation.
- [ ] Release object detection code.

## Acknowledgment
Thanks to these great repositories: [Pointcept](https://github.com/Pointcept/Pointcept), [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3), [Superpoint Transformer](https://github.com/drprojects/superpoint_transformer), [CroCo](https://github.com/naver/croco), [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Citation

If you find this project useful, please cite:
```
@inproceedings{yuelitept2026,
    title={{LitePT: Lighter Yet Stronger Point Transformer}},
    author={Yue, Yuanwen and Robert, Damien and Wang, Jianyuan and Hong, Sunghwan and Wegner, Jan Dirk and Rupprecht, Christian and Schindler, Konrad},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2026}
}
```