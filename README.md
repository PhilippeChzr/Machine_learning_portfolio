# Machine_learning_portfolio
This is a set of personal AI/Machine learning/Data science projects that I do in my free time.

## Contents
- ### Classification
	- [Titanic](https://github.com/PhilippeChzr/Machine_learning_portfolio/blob/main/Titanic)(**Data analysis**, **Random Forest**,  **Classification**, **Pytorch lightning**)
 The project comes from the well-known Titanic data-set. In [Titanic_analysis.ipynb](https://github.com/PhilippeChzr/Machine_learning_portfolio/blob/main/Titanic/Titanic_analysis.ipynb), we explore the data and try to select the more significant attributes. Then, we try to predict the attribute *survive* using a **Random Forest** in the notebook [Titanic_Estimation_RandomForest.ipynb](https://github.com/PhilippeChzr/Machine_learning_portfolio/blob/main/Titanic/Titanic_Estimation_RandomForest.ipynb). Finally, we create a neural network using ***Pytorch lightning*** when we find a model to realize the same prediction  [Titanic_Estimation_Pytorch.ipynb](https://github.com/PhilippeChzr/Machine_learning_portfolio/blob/main/Titanic/Titanic_Estimation_Pytorch.ipynb).

- ### Image classification
	- [Chest_X_Ray](https://github.com/PhilippeChzr/Machine_learning_portfolio/blob/main/Chext_X_Ray) (**Tensorflow**, **Image processing**, **Classification**)
The project is a Machine Learning project based on the Chest_X_Ray dataset, source: *https://data.mendeley.com/datasets/rscbjbr9sj/2*. The dataset is composed of X ray images of chests, each image is labeled *Normal* or *Pneumonia*. If the image is labeled *Pneumonia*, it can be *Bacteria* or *Virus*.
In this project, we are trying to classify images into one of three labels: normal, bacterial and viral.
In the first notebook [Chest_X_Ray_analysis.ipynb](https://github.com/PhilippeChzr/Machine_learning_portfolio/blob/main/Chext_X_Ray/Chest_X_Ray_analysis.ipynb), we try to improve the quality of each image, by detecting the rib cage and cropping the image around it. With the second  notebook [Chest_X_Ray_data.ipynb](https://github.com/PhilippeChzr/Machine_learning_portfolio/blob/main/Chext_X_Ray/Chest_X_Ray_download.ipynb), we check that dataset is well balanced and  we apply these image transformations to the entire dataset. In the third notebook [Chest_X_Ray_TF2_model.ipynb](https://github.com/PhilippeChzr/Machine_learning_portfolio/blob/main/Chext_X_Ray/Chest_X_Ray_TF2_model.ipynb), we create a CNN model with ***Tensorflow*** to predict the classification of each image and we evaluate it.

- ### Conditional GAN
	- [CGAN-mnist](https://github.com/PhilippeChzr/Machine_learning_portfolio/blob/main/CGAN-mnist) (**CGAN**, **Pytorch lightning**)
In this simple notebook [CGan.ipynb](https://github.com/PhilippeChzr/Machine_learning_portfolio/blob/main/CGAN-mnist/CGan.ipynb), we create a conditional GAN  using the MNIST dataset. The trained model can generate a image from a specific class.

	_Tools: Tensorflow, Pytorch lightning, PIL, OpenCV, Numpy, Pandas, matplotlib, seaborn_ 
