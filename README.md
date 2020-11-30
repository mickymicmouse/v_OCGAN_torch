### Evaluation setup
* Protocol 1
	* Train data (80% of training class), Valid data (10% of Train data), Test data (20% of training class + Test class)
* Protocol 2
	* Train data (Training class), Valid data (10% of Train data), Test data(Test data of all class)

### Experiment Order
* Train.py
    1. dataset - split data to training/validation/testing
    2. model - set OCGAN model
    3. dataloader - load data from data folder
    4. train, valid, test - model Train and Validation, Testing

* Test.py
    1. Test - model test, get AUROC Score


### Dataset Setting
* Original Dataset
	* datasource

                /class1
                /class2
                /class3
                /...

* Changed Dataset
    * dataroot
    
                /Training/1
                /Validation/1
                /Testing/1
                        /0


### Training and Validation, Testing


    python3 main.py --gpu-id 0 \
    --exp_name class1model \
    --datasource /home/data/tomato \
    --protocol 1 \
    --dataroot /home/data/tomato_split \
    --normal_class Yellow_leaf,Healthy,Late_blight,Septoria_leaf,Spider_mites,Bacterial_spot \
    --isize 64 \
    --epochs 200 \
    --batch 512 \
    --lr 0.0001 \
    --latent 16
    --checkpoint checkpoint/class1model \


### Testing


    python3 main.py --gpu-id 0 \
    --dataroot /home/data/tomato_split \
    --exp_name class1model \
    --normal_class Yellow_leaf,Healthy,Late_blight,Septoria_leaf,Spider_mites,Bacterial_spot \
    --protocol 1 \
    --isize 64 \
    --batch 512 \
    --checkpoint checkpoint/class1model \


### Log

* Show the train loss and test auc score at every epoch

### Picture

* Save the generated image and input image in result/exp_name folder



### Datasets
* MNIST
* FMNIST
* CIFAR10
* COIL100

### Architecture	
* Denoising Autoencoder
* Latent Discriminator
* Visual Discriminator
* Classifier

### Metric
* AUROC

### Model selection 
* Lowest validation loss epoch

### Model parameter Setup
| Parameter | Paper | Code | Description |
|:---------:|:---------:|:---------:|:---------:|
| loss  | MSE	| MSE	|	|
| optimizer	| Adam	| Adam	|	|
| btl_size  | 20    | opt  | latent vector size	|
| n_layers  | X    | 3  | number of layers in encoding	|
| epochs	| opt	| opt	| 	|
| batch size    | opt | opt	|	|
| img_size | opt	| opt	|	|
| Lambda | 10	| 10	|	|
| Beta1 | 0.5	| 0.5	| Adam optimizer parameter	|
| Ngf | 64	| 64	| number of filter of Generator	|
| Ndf | 12	| 12	| number of filter of Discriminator	|
| Noise Variance | 0.02	| 0.02	| Variance of input noise	|



### Paper
* https://arxiv.org/abs/1903.08550
* The author's implementation of *OCGAN* in MXNet is at [here](https://github.com/PramuPerera/OCGAN).
* Sub implementation of OCGAN in Pytorch is at [here](https://github.com/xiehousen/OCGAN-Pytorch)