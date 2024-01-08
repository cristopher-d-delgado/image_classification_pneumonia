# Pneumonia Image Classification
![x_ray_images](images/x_ray.jpg)

## Buisness Understanding 
Biomedical Imaging is important for clinical diagnostics to classify conditions a patient might have. In order to make these decisions domain knowldege is needed. The saying that a picture can describe a thousand words is true when doctors make there assessments but, it is limted only by the doctors perception of the biomedical images to classify conditions based on biomedical imaging. Creating deep learning models that can classify conditions can be useful in clinical practice to remove such high dependencies on a doctors domain knowledge. We have to acceptnNot all medical conditions are easy to classify based on medical imaging. Interpretation of an image is only limited to what someone knows and sees. Creating deep learning models that can classify conditions is crucial because deep learning models can learn what we humans can not visually see in biomedical imaging. Implementing such models in biomedical devices in the future can change the landscape of point of care diagnostics and clinical diagnostics.

## Data Understanding 
The X-ray images images used in this git repo are of pediatric patients. The classification is a binary case on whether a pateint has Pneumonia. The dataset comes from Kermany et al. on [Mendley](https://data.mendeley.com/datasets/rscbjbr9sj/3). The dataset on Kaggle is from the original source using Version 2 that was published on January 01, 2018. This dataset has gone through the trouble of image cleaning which is just getting rid of bad quality images. The training data consists of about 4000 images classified as Pneumonia and about 1500 images of Normal images (No Pneumonia). The testing data consists of about 400 Pneumonia images and about 250 Normal images. Unfortunatly the nature of this dataset originally contains 16 validation images. This is a small amount of images to truly monitor the validation loss. Instead a custom validation folder was made by incorportaing the 16 images in the val folder back into the train folder and randomly selecting 20% of the train set of each category. In this repository the custom data distribution was utilized.

### Original Data Distrubution
![class_distribution](images/original_data_dist.png)

### Custom Data Distribution
![modified_distribution](images/modified_data_dist.png)

Deep Learning uses information found in data which in this case are images. It is a type of machine learning that involves training artifical neural networks to perform tasks. In this notebook the task is to classify an image/condition as 'Normal' or 'Pneumonia'. The deep network learns to recognize patterns and features by adjusting its paramters based on the input data which are tensors that are derievd from images. 

The dataset does have an imbalance of images. This as a result may cause higher Recall/Sensitivity beacuse there are more pneumonia images to train on. This is the case because Pneumonia is our True positive case which as a result increaseing the the True Positive Rate which is Recall. We can use data augmentation to address this issue to create 'synthetic' images for the model to train on to have more availability to Negative cases which are normal images. Adding on, we can also attempt transder learning using the VGG19 pretrained model network to see what result we get on our test and training sets.

## Data Preparation

## Modeling 

## Evaluation