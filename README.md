# Classification Neural Network without TensorFlow or Keras

## Project Description

This project demonstrates the development of a classification neural network implemented entirely from scratch without using popular libraries like TensorFlow or Keras. The goal was to understand and manually implement key components of a neural network, including forward propagation, backward propagation, and model evaluation, while achieving practical classification performance.

* Input layer : 784 units or neurons
* Hidden layer : One layer of 10 units of neurons
* Output layer : 10 units or neurons

The project uses the MNIST dataset, a widely used dataset for image classification tasks. It consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. The dataset is divided into:

* Training Set: 60,000 images
* Test Set: 10,000 images

Each image is accompanied by a label corresponding to the digit it represents. The dataset was preprocessed to normalize pixel values and reshape images into a format suitable for neural network input.

## Project Workflow

* Forward Propagation

  - Linear Transformation

  - Activation Functions (ReLU, Softmax)

  - Weight and Bias Initialization

* Backward Propagation

  - Gradient Computation

  - Weight and Bias Updates

* Prediction

  - Training and Testing Performance Evaluation

* Evaluation

  - Metrics to assess model performance

## Model Performance

The project includes metrics to evaluate the training and testing performance of the implemented neural network. 

Key performance highlights for the training set includes:

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Iteration</th>
      <th>True Class</th>
      <th>TP</th>
      <th>FP</th>
      <th>TN</th>
      <th>FN</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Accuracy</th>
      <th>F1-Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>499</td>
      <td>0</td>
      <td>54</td>
      <td>716</td>
      <td>53361</td>
      <td>5869</td>
      <td>0.070130</td>
      <td>0.009117</td>
      <td>0.009117</td>
      <td>0.016136</td>
    </tr>
    <tr>
      <th>1</th>
      <td>499</td>
      <td>1</td>
      <td>327</td>
      <td>1893</td>
      <td>51365</td>
      <td>6415</td>
      <td>0.147297</td>
      <td>0.048502</td>
      <td>0.048502</td>
      <td>0.072975</td>
    </tr>
    <tr>
      <th>2</th>
      <td>499</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>54042</td>
      <td>5958</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>499</td>
      <td>3</td>
      <td>988</td>
      <td>11249</td>
      <td>42620</td>
      <td>5143</td>
      <td>0.080739</td>
      <td>0.161148</td>
      <td>0.161148</td>
      <td>0.107578</td>
    </tr>
    <tr>
      <th>4</th>
      <td>499</td>
      <td>4</td>
      <td>2586</td>
      <td>28815</td>
      <td>25343</td>
      <td>3256</td>
      <td>0.082354</td>
      <td>0.442657</td>
      <td>0.442657</td>
      <td>0.138872</td>
    </tr>
    <tr>
      <th>5</th>
      <td>499</td>
      <td>5</td>
      <td>0</td>
      <td>6</td>
      <td>54573</td>
      <td>5421</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>499</td>
      <td>6</td>
      <td>707</td>
      <td>5726</td>
      <td>48356</td>
      <td>5211</td>
      <td>0.109902</td>
      <td>0.119466</td>
      <td>0.119466</td>
      <td>0.114485</td>
    </tr>
    <tr>
      <th>7</th>
      <td>499</td>
      <td>7</td>
      <td>998</td>
      <td>5897</td>
      <td>47838</td>
      <td>5267</td>
      <td>0.144743</td>
      <td>0.159298</td>
      <td>0.159298</td>
      <td>0.151672</td>
    </tr>
    <tr>
      <th>8</th>
      <td>499</td>
      <td>8</td>
      <td>0</td>
      <td>2</td>
      <td>54147</td>
      <td>5851</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>499</td>
      <td>9</td>
      <td>4</td>
      <td>32</td>
      <td>54019</td>
      <td>5945</td>
      <td>0.111111</td>
      <td>0.000672</td>
      <td>0.000672</td>
      <td>0.001337</td>
    </tr>
  </tbody>
</table>
</div>

Additionnaly, here are some graph displaying the evoluation of the mean and maximum accuracy of the training set.


![ce2b53d1-a709-4b9d-9e0a-d3c91473bbbe](https://github.com/user-attachments/assets/38be2427-73d4-40c5-b089-c89b59c69525)

![77d247ca-1d2d-4771-9787-098da07d016f](https://github.com/user-attachments/assets/e486e06d-6b68-4fa2-9a04-de0378e8a3f0)

## Dependencies

This project was done on Python 3.12.2. However, you can probably do it on most python versions.
To run this project, the following libraries are required:

* NumPy

* Matplotlib (optional, for visualizations)

## Usage

To reproduce the results or adapt the neural network implementation for your own dataset:

* Clone the repository.

* Ensure all dependencies are installed.

* Run the Jupyter Notebook script.ipynb to execute the neural network pipeline step by step.

## Results

The project concludes with an evaluation of the network’s accuracy, precision and recall on both the training and testing datasets, demonstrating its lacking ability to classify inputs correctly. 
Further analysis of errors or potential improvements is discussed in the notebook. The most notable improvement being the depth of the model.
