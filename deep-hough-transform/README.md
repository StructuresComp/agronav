<h1 style="align: center; color: #159957">Deep Hough Transform for Semantic Line Detection</h1>

Link to original Github of Deep Hough Transform (https://github.com/Hanqer/deep-hough-transform).
Link to paper "Deep Hough Transform for Semantic Line Detection" (ECCV 2020, PAMI 2021).
[arXiv2003.04676](https://arxiv.org/abs/2003.04676) | [Online Demo](http://mc.nankai.edu.cn/dht) | [Project page](http://mmcheng.net/dhtline) | [New dataset](http://kaizhao.net/nkl) | [Line Annotator](https://github.com/Hanqer/lines-manual-labeling)

### Requirements
``` 
numpy
scipy
opencv-python
scikit-image
pytorch>=1.0
torchvision
tqdm
yml
POT
deep-hough
```

To install deep-hough, run the following commands.

```sh
cd deep-hough-transform
cd model/_cdht
python setup.py build 
python setup.py install --user
```

### Training on Custom AgroNav Data
1. Download the AgroNav_LineDetection dataset from [here](https://drive.google.com/file/d/1MPaQVXCWcpGZT5Kfe3fOYBoR3PYghjt9/view?usp=sharing) and extract to `data/` directory. The dataset contains images and ground truth annotations of the semantic lines. The images are the outputs of the semantic segmentation model. Each image contains a pair of semantic lines.

2. Run the following lines for data augmentation and generation of the parametric space labels.
```sh
cd deep-hough-transform
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

### Inference
1. Download pre-trained checkpoint [here](https://drive.google.com/file/d/1Q3s_QKUJiiCGibNzF44hQu8jfBK_Bxor/view?usp=sharing). Move the file to /checkpoint/

2. Move images to data/inference/input/

3. Run the following script to obtain a list of filenames of the testing data. This creates a .txt file with the filenames inside data/inference/
```sh
python data/inference/extractFilenameList.py
```

4. Specify the testing data paths in config.yml.

5. Run the following script for inference
```sh
python forward.py --model ./checkpoint/model_best.pth --tmp ./data/inference/output/
```

6. Visualize the centerlines.
```sh
python data/inference/visualizeCenterline.py 
```
The visualizations are saved in ./data/inference/centerline_visualized

<!-- ### Citation
If our method/dataset are useful to your research, please consider to cite us:
```
@article{zhao2021deep,
  author    = {Kai Zhao and Qi Han and Chang-bin Zhang and Jun Xu and Ming-ming Cheng},
  title     = {Deep Hough Transform for Semantic Line Detection},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year      = {2021},
  doi       = {10.1109/TPAMI.2021.3077129}
}
```
```
@inproceedings{eccv2020line,
  title={Deep Hough Transform for Semantic Line Detection},
  author={Qi Han and Kai Zhao and Jun Xu and Ming-Ming Cheng},
  booktitle={ECCV},
  pages={750--766},
  year={2020}
}
```

### License
This project is licensed under the [Creative Commons NonCommercial (CC BY-NC 3.0)](https://creativecommons.org/licenses/by-nc/3.0/) license where only
non-commercial usage is allowed. For commercial usage, please contact us. -->

