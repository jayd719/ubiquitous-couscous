<main>
<div style="text-align: center;">
<h1 style="font-size: xx-large;">K-Nearest Neighbors (K-NN) Algorithm</h1>
<h1>Implementation and Analysis</h1>
</div>


<div id="index"></div>

<h2>Overview</h2>
<p>This project is about building a <strong>k-Nearest Neighbors (k-NN) classifier</strong> from
scratch
using
Python. The k-NN algorithm is a simple machine learning method used for
<strong>classification</strong>
tasks.
Instead of learning a mathematical model, it memorizes the training data and makes
predictions by
looking at the
<strong>k closest data points</strong>.
</p>
<h4><strong>How It Works:</strong></h4>
<ol>
<li>The algorithm calculates the <strong>distance</strong> between a new data point and all
training
examples.
</li>
<li>It selects the <strong>k nearest neighbors</strong> (most similar examples).</li>
<li>The class (label) with the most votes among these neighbors is assigned to the new data
point.</li>
</ol>
<h2>Implementation</h2>

<p>K-Nearest Neighbors (K-NN) is a simple method for classification. It works by comparing new data points with
existing ones and classifying them based on similarity. The main steps include:</p>
<ul>
<li><strong>Distance Calculation:</strong> The algorithm measures how close data points are using Euclidean
or
Manhattan distance.</li>
<li><strong>Fit Method:</strong> This step stores the training data for later use.</li>
<li><strong>Predict Method:</strong> It finds the <code>k</code>-nearest points to the new data and assigns
the
most common class.</li>
<li><strong>Score Method:</strong> This calculates how well the model performs by checking its accuracy.
</li>
</ul>

<h2>Data Preparation and Preprocessing</h2>
<p>Before using K-NN, the data needs to be prepared. The dataset includes:</p>
<ul>
<li><strong>Training Inputs:</strong> <code>X_train</code> (features of training samples)</li>
<li><strong>Training Labels:</strong> <code>y_train</code> (correct class for training samples)</li>
<li><strong>Test Inputs:</strong> <code>X_test</code> (features of test samples)</li>
<li><strong>Test Labels:</strong> <code>y_test</code> (correct class for test samples)</li>
</ul>
<p>To make sure all features contribute equally, the <code>StandardScaler</code> from Scikit-learn is used to
scale
them.</p>

<h2>Testing</h2>
<div style="text-align: center; padding-bottom: 10%;">
<img width="75%" class=""
src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*C1AAnbD-OJFemKra.png" alt="">
</div>
<p>The testing was performed using K-Fold Cross Validation, where the dataset
(X,y) was divided into 10 equal parts (folds). The model was trained on 9 of these parts and tested on the
remaining
part. This process was repeated 10 times, ensuring that each part of the data served as a test set once.

To determine the optimal hyperparameter, different values of K (ranging from 1 to 30) were tested for the
K-Nearest
Neighbors (KNN) model. For each value of
K, the model’s accuracy was computed across all folds, and the average accuracy was recorded.

The best value of
K was identified as the one that yielded the highest average accuracy. The results were stored in two lists:
one
containing the tested
K
K values and the other containing their corresponding average accuracies. These results helped in selecting
the most
effective KNN model for the dataset.</p>

<h2>Performance Analysis</h2>
<p>The results below were obtained by testing the K-Nearest Neighbors (KNN) algorithm on a modified version of
the MNIST
dataset. This dataset includes only images of the digits 5 and 6. The test aimed to evaluate how well KNN
can classify
these two digits</p>
<img width="100%" src="plot-one.png" alt="">
<p>The accuracy of K-NN depends on the choice of <code>k</code>:</p>
<ul>
<li><strong>Small k Values:</strong> The model performs poorly at <code>k = 1</code> and <code>k = 2</code>
with
<strong>74.25%</strong> accuracy. Small <code>k</code> makes the model sensitive to noise.
</li>
<li><strong>Moderate k Values:</strong> When <code>k</code> is <code>3</code> or <code>4</code>, accuracy
improves to <strong>78.5%</strong>, as the effect of noise reduces.</li>

<li><strong>Large k Values:</strong> The accuracy peaks at <code>k = 15</code> with <strong>79.51%</strong>.
However, increasing <code>k</code> too much causes loss of detail and higher computation time.</li>
</ul>

<h2>Additional Tests</h2>
<p>To further evaluate the performance of K-NN, additional tests were conducted using various datasets from
Scikit-learn:</p>
<ul>
<li><strong>Iris Dataset:</strong> A well-known dataset for classification. </li>
<li><strong>Digits Dataset:</strong> Handwritten digit recognition.</li>
<li><strong>Wine Dataset:</strong> Classification of different wine types.</li>
<li><strong>Breast Cancer Dataset:</strong> Medical diagnosis of breast cancer.</li>
<li><strong>Diabetes Dataset:</strong> Predicting diabetes progression.</li>
<li><strong>California Housing Dataset:</strong> Regression task for housing prices.</li>
</ul>

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
<img width="100%" src="IRIS.png" alt="">
<img width="100%" src="DIGITS.png" alt="">
<img width="100%" src="BREAST CANCER.png" alt="">
<img width="100%" src="WINE.png" alt="">
</div>



<h2>Conclusion</h2>
<p>The K-NN algorithm is effective for classification. Choosing the right <code>k</code> is key to its
performance.
</p>


</main>