# DLS project 2022

This is PyTorch implementation of the [CycleGan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) network developed as the [Deep Learning School](https://www.dlschool.org/) (DLS) final project. The goal of the project was to create the CycleGan network for image translation between two species of birds - Parus major and Sparrow. For this purpose based on the [original paper](https://arxiv.org/abs/1703.10593) (see also [Ref](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html)), the CycleGan model and a simplified training procedure for it were implemented. A buffer for images, the possibility to load and save checkpoints, linear decay of the learning rate, and generation of images from a test dataset were realized. The code was preliminarily tested on the zebra-to-horse task and give results close (visually) to the ones of the [original paper](https://arxiv.org/abs/1703.10593).

# Datasets
For Parus major <-> Sparrow image translation task two datasets are used. The first one contains 129 images of the "Parus major" class and 132 images of the "Sparrow" class. The corresponding images were downloaded from [images.cv](https://images.cv ) and is available in the current repository (datasets/small_dataset.zip). However, since this dataset is too small to achieve an acceptable good quality of results, for the main training procedure of the model another (second) dataset was used. This second dataset is a substantial extension of the first dataset by including stock (free license) images of both classes. The corresponding dataset (not provided here) consists of 632 images of the "Parus major" class and 633 images of the "Sparrow" class. For both datasets, the images were scaled to 256x256 pixels.

# Usage 

1. Clone repository to local:

```git clone https://github.com/vladimirprotsenko/DLS__project_2022.git```

2. Run the script to prepare the dataset:  

```bash scripts/prepare_dataset.sh``` 

or 

```chmod +x scripts/prepare_dataset.sh```

```scripts/prepare_dataset.sh```


Note that the "small dataset" (see Datasets section) is used by default. To use your own dataset change `dataset_path` variable (currently `dataset_path="datasets/small_dataset.zip"`) in scripts/prepare_dataset.sh file according to the name (and the path) of your dataset .zip archive. Another possibility is to directly specify the path to a dataset via the `--path_to_data` key when running train.py or generate.py. **Data for each class should be separated to train and test subfolders, e.g (trainX,trainY,testX,testY) or (trainA,trainB,testA,testB)).**

3. Train/generation:

* To train model run:

```python train.py [options]```

For more information about the available options use:

```python train.py --help ```     

* To generate images from test dataset run:

```python generate.py --load_epoch N [options]```

Here N is a checkpoint epoch to load from. For more information about the available options use:

```python generate.py --help ```  

Note, that in the case of [Google Colab](https://colab.research.google.com/) command `X` should be `!X`, e.g. `!python train.py --help ` (or use %%shell).

# Results

The model was trained for 300 epochs with batch size=1. The learning rate was constant (lr=0.0002) for the first 200 epochs and then decayed linearly to zero. Weights of the cycle consistency losses and the identity losses were 10 and 0.05, respectively. Image buffer with maximum depth of 50 images was used. 

As can be seen from the images below, the network was able to learn the qualitative patterns of translation between different image domains. However, I have not been able to achieve a significant quality of generated images within the above-stated training procedure. Perhaps this is due to the small size of the dataset and its poor quality (in particular, the presence of a significant percentage of duplicate images and a strong variation in the color of Parus major birds depending on the halo and sex). It also seems that the number of training epochs should be somewhat larger. 


## Examples: real image (left) to generated image (right)

More results (including failed cases) are available in the "results" folder of this repository.  


![ex1](https://github.com/vladimirprotsenko/DLS_project_2022/blob/main/results/imgs_results/XtoYandYtoX_epoch_itr_483.png?raw=true)

![ex2](https://github.com/vladimirprotsenko/DLS_project_2022/blob/main/results/imgs_results/XtoYandYtoX_epoch_itr_486.png?raw=true)

![ex3](https://github.com/vladimirprotsenko/DLS_project_2022/blob/main/results/imgs_results/XtoYandYtoX_epoch_itr_369.png?raw=true)

![ex2](https://github.com/vladimirprotsenko/DLS_project_2022/blob/main/results/imgs_results/XtoYandYtoX_epoch_itr_329.png?raw=true)

![ex2](https://github.com/vladimirprotsenko/DLS_project_2022/blob/main/results/imgs_results/XtoYandYtoX_epoch_itr_268.png?raw=true)

![ex2](https://github.com/vladimirprotsenko/DLS_project_2022/blob/main/results/imgs_results/XtoYandYtoX_epoch_itr_228.png?raw=true)



