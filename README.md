
---

## Agronav: Autonomous Navigation Framework for Agricultural Robots and Vehicles using Semantic Segmentation and Semantic Line Detection

* This repository contains the codebase for our work presented at the [4th International Workshop on
AGRICULTURE-VISION: CHALLENGES & OPPORTUNITIES FOR COMPUTER VISION IN AGRICULTURE](https://https://www.agriculture-vision.com/) from CVPR 2023.
* The link to the original paper can be found [here](https://openaccess.thecvf.com/content/CVPR2023W/AgriVision/papers/Panda_Agronav_Autonomous_Navigation_Framework_for_Agricultural_Robots_and_Vehicles_Using_CVPRW_2023_paper.pdf).
* Our work uses [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) to train our semantic segmentation model and [Deep Hough Transform](https://github.com/Hanqer/deep-hough-transform) for semantic line detection.


<p align="center">
<image src= "./figures/pipeline.png" vspace="10">
<br>
<em> Pipeline of the Agronav framework </em>


---

### Updates
* 06/22/2023: Revised instructions, tested image inference code.

---

## Instructions

---

### Dependencies

This code has been tested on Python 3.8.

1. After creating a virtual environment (Python 3.8), install pytorch and cuda package.
    
    Using `conda`:
    ```bash
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    ```
    Using `pip`:
    ```bash
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    ```
2. Install `mmcv`
    ```bash
    pip install mmcv-full==1.7.1
    cd segmentation
    pip install -v -e .
    ```
3. Install other dependencies
    ```bash
   conda install numpy scipy scikit-image pqdm -y
   pip install opencv-python yml POT pudb -y
    ```
4. Install `deep-hough-transform`
    ```bash
    cd ../lineDetection
    cd model/_cdht
    python setup.py build 
    python setup.py install --user
    ```
---

### Training the Semantic Segmentation Model

1. Download the Agroscapes Segmentation dataset from  [here](https://drive.google.com/drive/folders/1XfvWrEmAVhW9r6PF-46aBSf32KLaqshq?usp=sharing) and extract the images and labels to `data/agronav/images` and `data/agronav/labels` respectively.

2. Run `python train-agronav.py` to start training. Although before that check the file `agronav.py` for the configuration of training (`cfg`). You might want to edit the python file for cfg based on your training model. Accordingly edit `cfg.load_from` and `cfg.work_dir`, for your checkpoint and output directory respectively. 

---

### Training the Semantic Line Detection Model

1. Download the AgroNav_LineDetection dataset from [here](https://drive.google.com/file/d/1MPaQVXCWcpGZT5Kfe3fOYBoR3PYghjt9/view?usp=sharing) and extract to `data/` directory. The dataset contains images and ground truth annotations of the semantic lines. The images are the outputs of the semantic segmentation model. Each image contains a pair of semantic lines.

2. Run the following lines for data augmentation and generation of the parametric space labels.
```sh
cd lineDetection
python data/prepare_data_NKL.py --root './data/agroNav_LineDetection' --label './data/agroNav_LineDetection' --save-dir './data/training/agroNav_LineDetection_resized_100_100' --fixsize 400 
```

3. Run the following script to obtain a list of filenames of the training data. 
```sh
python data/extractFilenameList.py
```
This creates a .txt file with the filenames inside /training. Divide the filenames into train and validation data.
```sh
agroNav_LineDetection_train.txt
agroNav_LineDetection_val.txt
```

4. Specify the training and validation data paths in config.yml.

5. Train the model.
```sh
python train.py
```
---

### Inference
1. Download the pre-trained checkpoints for semantic segmentation [[MobileNetV3](https://drive.google.com/file/d/1CEL6JfLZbvZyaB0TL-cYeC9JvQkhsGDI/view?usp=sharing), [HRNet](https://drive.google.com/file/d/1oTbwQmOLEcL5ix4sKpANRvZX_-AyInLG/view?usp=sharing), [ResNest](https://drive.google.com/file/d/1sGZNJiUy9NyaQPFf3kFVuzlOI5-E4_xF/view?usp=sharing)]. Move the downloaded file to `./segmentation/checkpoint/`.
2. Download the pre-trained checkpoints for semantic line detection [here](https://drive.google.com/file/d/1Q3s_QKUJiiCGibNzF44hQu8jfBK_Bxor/view?usp=sharing). Move the download file to `../lineDetection/checkpoint/`.
3. Move inference images to `./inference/input`
4. Run the following command to perform end-to-end inference on the test images. End-to-end inference begins with a raw RGB image, and visualizes the centerlines.
    ```bash
   python e2e_inference_image.py
    ```
5. The final results with the centerlines will be saved in `./inference/output_ceterline`, the intermediate results are saved in `./inference/temp` and `./inference/output`.

*To run the semantic segmentation and line detection models independently, use `./segmentation/inference_image.py` and `./lineDetection/inference.py`*.

---

## Citation
If found helpful, please consider citing our work:

```bash
@InProceedings{Panda_2023_CVPR,
    author    = {Panda, Shivam K. and Lee, Yongkyu and Jawed, M. Khalid},
    title     = {Agronav: Autonomous Navigation Framework for Agricultural Robots and Vehicles Using Semantic Segmentation and Semantic Line Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {6271-6280}
}
```


