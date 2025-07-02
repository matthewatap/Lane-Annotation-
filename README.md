# Lane Detection and Classification System

This project is a modified version of YOLOPv2, enhanced with improved lane detection and classification capabilities. The system can detect, classify, and annotate different types of lane markings in real-time video feeds.

## Features

- **Accurate Lane Detection**: Uses YOLOPv2 backbone with improved aspect ratio handling
- **Lane Classification**: Identifies multiple lane types:
  - White solid lines
  - Yellow solid lines
  - White dashed lines
  - Yellow dashed lines
  - Road edges
- **Real-time Processing**: Processes video feeds in real-time with visual overlay
- **Debug Visualization**: Includes alignment checking and debug output options
- **Aspect Ratio Preservation**: Maintains proper image geometry through letterboxing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/matthewatap/Lane-Annotation-.git
cd Lane-Annotation-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the YOLOPv2 weights and place them in the `weights` directory:
```bash
mkdir -p weights
# Download yolopv2.pt and place it in the weights directory
```

## Usage

### Basic Usage

Run lane detection on a video file:
```bash
python lane_classifier.py --source path/to/your/video.mp4 --output path/to/output.mp4 --weights weights/yolopv2.pt
```

### Debug Mode

To run with debug visualization:
```bash
python lane_classifier.py --source path/to/your/video.mp4 --output path/to/output.mp4 --weights weights/yolopv2.pt --debug
```

### Parameters

- `--source`: Path to input video file
- `--output`: Path to save output video
- `--weights`: Path to YOLOPv2 weights file
- `--device`: Device to run on ('cuda' or 'cpu')
- `--debug`: Enable debug mode with visualization

## Project Structure

- `lane_classifier.py`: Main lane detection and classification implementation
- `demo.py`: Original YOLOPv2 demo script
- `utils/`: Utility functions and helpers
- `weights/`: Directory for model weights
- `data/`: Input and output video directories

## Output Format

The system generates:
1. Processed video with lane overlays
2. Lane type labels
3. Debug visualizations (when enabled):
   - Original frame
   - Mask alignment view
   - Final result with overlays

## Improvements

This version includes several improvements over the base YOLOPv2:
1. Proper aspect ratio handling through letterboxing
2. Enhanced lane classification
3. Improved visualization and overlay system
4. Debug mode for development and troubleshooting

## License

This project is based on YOLOPv2 and follows its original licensing terms.

## Acknowledgments

- Original YOLOPv2 implementation by CAIC-AD
- Modified and enhanced by matthewatap

<div align="center">
<h1> YOLOPv2:rocket:: Better, Faster, Stronger for Panoptic driving Perception </h1>
<!-- <--!span><font size="5", > Efficient and Robust 2D-to-BEV Representation Learning via Geometry-guided Kernel Transformer
</font></span> -->

  Cheng Han, Qichao Zhao, Shuyi Zhang, Yinzi Chen, Zhenlin Zhang, Jinwei Yuan
<!-- <a href="https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN">Jinwei Yuan</a> -->
<div><a href="https://arxiv.org/abs/2208.11434">[YOLOPv2 arXiv Preprint]</a></div> 

</div>

## News
* `August 30, 2022`: We've released the inference code / trained  <a href="https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt">model</a> and published [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/hank1996/yolopv2)[web demo](https://huggingface.co/spaces/hank1996/yolopv2), just enjoy it !

* `August 24, 2022`: We've released the tech report for **YOLOPv2**. This work is still in progress and code/models are coming soon. Please stay tuned! ☕️
<!-- * `August 26, 2022`: We've upload the model for **YOLOPv2**. This version support for model training, validation and prediction. -->

## Introduction

:grin:We present an excellent multi-task network based on YOLOP:blue_heart:,which is called **YOLOPv2: Better, Faster, Stronger for Panoptic driving Perception**.
The advantages of **YOLOPv2** can be summaried as below:
* **Better**:clap:: we proposed the end-to-end perception network which possess better feature extraction backbone, better bag-of-freebies were developed for dealing with the training process.
* **Faster**:airplane:: we employed more efficient ELAN structures to achieve reasonable memory allocation for our  model. 
* **Stronger**:muscle:: the proposed model has stable network design and has powerful robustness for adapting to various scenarios .

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolopv2-better-faster-stronger-for-panoptic/traffic-object-detection-on-bdd100k)](https://paperswithcode.com/sota/traffic-object-detection-on-bdd100k?p=yolopv2-better-faster-stronger-for-panoptic)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolopv2-better-faster-stronger-for-panoptic/drivable-area-detection-on-bdd100k)](https://paperswithcode.com/sota/drivable-area-detection-on-bdd100k?p=yolopv2-better-faster-stronger-for-panoptic)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolopv2-better-faster-stronger-for-panoptic/lane-detection-on-bdd100k)](https://paperswithcode.com/sota/lane-detection-on-bdd100k?p=yolopv2-better-faster-stronger-for-panoptic)

## Results
We used the BDD100K as our datasets,and experiments are run on **NVIDIA TESLA V100**.


### Web Demo

- Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces/hank1996/yolopv2) using Gradio. Try out the Web Demo !

### Visualization
model : trained on the BDD100k dataset and test on T3CAIC camera.
<td><img src=data/demo/together_video.gif/></td>
  
  



### Model parameter and inference speed
|        Model       |    Size     |   Params   |  Speed (fps) |
|:------------------:|:-----------:|:----------:|:-----------:|
|       `YOLOP`      |     640     |    7.9M    |     49      |
|     `HybridNets`   |     640     |    12.8M   |     28      |
|    **`YOLOPv2`**   |     640     |  **38.9M** |   **91 (+42)** :arrow_double_up:   |


### Traffic Object Detection Result
<table>
<tr><th>Result </th><th>Visualization</th></tr>
<tr><td>
  

|        Model       |     mAP@0.5 (%)   |Recall (%)  |
|:------------------:|:------------:|:---------------:|
|     `MultiNet`     |        60.2      |   81.3     |  
|      `DLT-Net`     |        68.4      |  89.4     |
|   `Faster R-CNN`   |          55.6      | 77.2     |
|      `YOLOv5s`     |         77.2      | 86.8     |
|       `YOLOP`      |        76.5      | 89.2     |   
|     `HybridNets`   |          77.3      | **92.8**   | 
|    **`YOLOPv2`**   |       **83.4(+6.1)**:arrow_double_up:    |   91.1(-1.7)  :arrow_down:   |

</td><td>

<!-- ### Visualization -->
<img src="data/demo/veh3.jpg" width="100%" align='right'/>
<!--   <img src="images/veh2.jpg" width="50%" /> -->

</td></tr> </table>

### Drivable Area Segmentation
<table>

<tr><th>Result </th><th>Visualization</th></tr>
<tr><td>

|       Model      | Drivable mIoU (%) |——:relaxed:——|
|:----------------:|:-----------------:|---:|
|    `MultiNet`    |        71.6       |  |   
|     `DLT-Net`    |        71.3       || 
|     `PSPNet`     |        89.6       || 
|      `YOLOP`     |        91.5       || 
|     `HybridNets` |        90.5       || 
|     `YOLOPv2`    |   **93.2(+1.7)** :arrow_up:  |  | 

</td><td>

<!-- ### Visualization -->
<img src="data/demo/fs3.jpg" width="100%" align='right'/>
<!--   <img src="images/fs2.jpg" width="50%" /> -->

</td></tr> </table>

### Lane Line Detection
<table>
<tr><th>Result </th><th>Visualization</th></tr>
<tr><td>

|      Model       | Accuracy (%) | Lane Line IoU (%) |
|:----------------:|:------------:|:-----------------:|
|      `Enet`      |     34.12    |       14.64       |
|      `SCNN`      |     35.79    |       15.84       |
|    `Enet-SAD`    |     36.56    |       16.02       |
|      `YOLOP`     |     70.5     |        26.2       |
|   `HybridNets`   |     85.4     |        **31.6**     |
|    **`YOLOPv2`** |   **87.3(+1.9)**:arrow_up:   |      27.2(-4.4) :arrow_down: |
  
</td><td>

<!-- ### Visualization -->
<img src="data/demo/lane3.jpg" width="100%" align='right' />
<!--   <img src="images/lane1.jpg" width="50%" /> -->

</td></tr> </table>

### Day-time and Night-time visualization results
<div align = 'None'>
  <a href="./">
  <img src="data/demo/all3.jpg" width="45%" />
  <img src="data/demo/all2.jpg" width="45%" />
  <img src="data/demo/night1.jpg" width="45%" />
  <img src="data/demo/night2.jpg" width="45%" />
  </a>
</div>



## Models

You can get the model from <a href="https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt">here</a>.


### Demo Test

We provide two testing method.You can store the image or video.

```shell
python demo.py  --source data/example.jpg
```

<!-- ## Usage
coming soon. -->


### Third Parties Resource  

* YOLOPv2 NCNN C++ Demo: [YOLOPv2-ncnn](https://github.com/FeiGeChuanShu/YOLOPv2-ncnn) from [FeiGeChuanShu](https://github.com/FeiGeChuanShu/)
* YOLOPv2 ONNX and OpenCV DNN Demo: [yolopv2-opencv-onnxrun-cpp-py](https://github.com/hpc203/yolopv2-opencv-onnxrun-cpp-py) from [hpc203](https://github.com/hpc203/)


## License

YOLOPv2 is released under the [MIT Licence](LICENSE).

<!-- ## Citation
If you find YOLOPv2 is useful in your research or applications, please consider giving us a star &#127775; and citing it by the following BibTeX entry.
```bibtex
@article{GeokernelTransformer,
  title={Efficient and Robust 2D-to-BEV Representation Learning via Geometry-guided Kernel Transformer},
  author={Chen, Shaoyu and Cheng, Tianheng and Wang, Xinggang and Meng, Wenming and Zhang, Qian and Liu, Wenyu},
  journal={arXiv preprint arXiv:2206.04584},
  year={2022}
}
``` -->
