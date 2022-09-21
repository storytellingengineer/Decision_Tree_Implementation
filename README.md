# Decision_Tree_Implementation
<a class="anchor" id="0"></a>
Hello friends,

In this repository, I have used a Decision Tree Classifier to predict the safety of the car. I built two models, one with criterion `gini index` and another one with criterion `entropy`. I implemented DecisionTreeClassifier with Python and Scikit-Learn.

So, let's get started.

<a class="anchor" id="0.1"></a>
# **Table of Contents**

1.	[Introduction to Decision Tree algorithm](#1)
2.	[Classification and Regression Trees](#2)
3.	[Decision Tree algorithm terminology](#3)
4.	[Decision Tree algorithm intuition](#4)
5.	[Attribute selection measures](#5)
    - 5.1 [Information gain](#5.1)
    - 5.2 [Entropy](#5.2)
    - 5.3 [Gini index](#5.3)
6.	[Overfitting in Decision-Tree algorithm](#6)
7.	[Import libraries](#7)
8.	[Import dataset](#8)
9.	[Exploratory data analysis](#9)
10.	[Declare feature vector and target variable](#10)
11.	[Split data into separate training and test set](#11)
12.	[Feature engineering](#12)
13.	[Decision Tree classifier with criterion gini-index](#13)
14.	[Decision Tree classifier with criterion entropy](#14)
15.	[Confusion matrix](#15)
16.	[Classification report](#16)
17.	[Results and conclusion](#17)
18. [References](#18)

# **1. Introduction to Decision Tree algorithm** <a class="anchor" id="1"></a>
[Table of Contents](#0.1)

A Decision Tree algorithm is one of the most popular machine learning algorithms. It uses a tree like structure and their possible combinations to solve a particular problem. It belongs to the class of supervised learning algorithms where it can be used for both classification and regression purposes. 

A decision tree is a structure that includes a root node, branches, and leaf nodes. Each internal node denotes a test on an attribute, each branch denotes the outcome of a test, and each leaf node holds a class label. The topmost node in the tree is the root node. 

We make some assumptions while implementing the Decision-Tree algorithm. These are listed below:-

1. At the beginning, the whole training set is considered as the root.
2. Feature values need to be categorical. If the values are continuous then they are discretized prior to building the model.
3. Records are distributed recursively on the basis of attribute values.
4. Order to placing attributes as root or internal node of the tree is done by using some statistical approach.

I will describe Decision Tree terminology in later section.

# **2. Classification and Regression Trees (CART)** <a class="anchor" id="2"></a>
[Table of Contents](#0.1)

Nowadays, Decision Tree algorithm is known by its modern name **CART** which stands for **Classification and Regression Trees**. Classification and Regression Trees or **CART** is a term introduced by Leo Breiman to refer to Decision Tree algorithms that can be used for classification and regression modeling problems.

The CART algorithm provides a foundation for other important algorithms like bagged decision trees, random forest and boosted decision trees. In this repo, I will solve a classification problem. So, I will refer the algorithm also as Decision Tree Classification problem. 

# **3. Decision Tree algorithm terminology** <a class="anchor" id="3"></a>
[Table of Contents](#0.1)

- In a Decision Tree algorithm, there is a tree like structure in which each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label. The paths from the root node to leaf node represent classification rules.
- We can see that there is some terminology involved in Decision Tree algorithm. The terms involved in Decision Tree algorithm are as follows:-
    1. ## **Root Node**
        - It represents the entire population or sample. This further gets divided into two or more homogeneous sets.
    2. ## **Splitting**
        - It is a process of dividing a node into two or more sub-nodes.
    3. ## **Decision Node**
        - When a sub-node splits into further sub-nodes, then it is called a decision node.
    4. ## **Leaf/Terminal Node**
        - Nodes that do not split are called Leaf or Terminal nodes.
    5. ## **Pruning**
        - When we remove sub-nodes of a decision node, this process is called pruning. It is the opposite process of splitting.
    6. ## **Branch/Sub-Tree**
        - A sub-section of an entire tree is called a branch or sub-tree.
    7. ## **Parent and Child Node**
        - A node, which is divided into sub-nodes is called the parent node of sub-nodes where sub-nodes are the children of a parent node. 

The above terminology is represented clearly in the following diagram:-
![image](https://user-images.githubusercontent.com/35486320/191093651-0f312592-8f8b-48d8-a51b-cee3ba578e47.png)

# **4. Decision Tree algorithm intuition** <a class="anchor" id="4"></a>
[Table of Contents](#0.1)

The Decision-Tree algorithm is one of the most frequently and widely used supervised machine learning algorithms that can be used for both classification and regression tasks. The intuition behind the Decision-Tree algorithm is very simple to understand.

The Decision Tree algorithm intuition is as follows:-

1.	For each attribute in the dataset, the Decision-Tree algorithm forms a node. The most important attribute is placed at the root node. 
2.	For evaluating the task in hand, we start at the root node and we work our way down the tree by following the corresponding node that meets our condition or decision.
3.	This process continues until a leaf node is reached. It contains the prediction or the outcome of the Decision Tree.

# **5. Attribute selection measures** <a class="anchor" id="5"></a>
[Table of Contents](#0.1)

The primary challenge in the Decision Tree implementation is to identify the attributes which we consider as the root node and each level. This process is known as the **attributes selection**. There are different attributes selection measure to identify the attribute which can be considered as the root node at each level.

There are 2 popular attribute selection measures. They are as follows:-

- **Information gain**
- **Gini index**

While using **Information gain** as a criterion, we assume attributes to be categorical and for **Gini index** attributes are assumed to be continuous. These attribute selection measures are described below.

## **5.1 Information gain** <a class="anchor" id="5.1"></a>
[Table of Contents](#0.1)

By using information gain as a criterion, we try to estimate the information contained by each attribute. To understand the concept of Information Gain, we need to know another concept called **Entropy**. 

## **5.2 Entropy** <a class="anchor" id="5.2"></a>
[Table of Contents](#0.1)

Entropy measures the impurity in the given dataset. In Physics and Mathematics, entropy is referred to as the randomness or uncertainty of a random variable X. In information theory, it refers to the impurity in a group of examples. **Information gain** is the decrease in entropy. Information gain computes the difference between entropy before split and average entropy after split of the dataset based on given attribute values. 

Entropy is represented by the following formula:-

![image](https://user-images.githubusercontent.com/35486320/191333640-7cd146fa-5068-496c-a9bf-d18f21ab9ef1.png)

Here, `c` is the number of classes and `pi` is the probability associated with the $i^{th}$ class. 

## **5.3 Gini index** <a class="anchor" id="5.3"></a>
[Table of Contents](#0.1)

Another attribute selection measure that **CART (Categorical and Regression Trees)** uses is the **Gini index**. It uses the Gini method to create split points. 

Gini index can be represented with the following diagram:-\

![image](https://user-images.githubusercontent.com/35486320/191420957-421cbc60-80e6-4b4f-8b5b-be7434df08fc.png)

Here, again `c` is the number of classes and `pi` is the probability associated with the ith class.

Gini index says, if we randomly select two items from a population, they must be of the same class and probability for this is 1 if the population is pure.

It works with the categorical target variable `Success` or `Failure`. It performs only binary splits. The higher the value of Gini, higher the homogeneity. CART (Classification and Regression Tree) uses the Gini method to create binary splits.

Steps to Calculate Gini for a split

1.	Calculate Gini for sub-nodes, using formula sum of the square of probability for success and failure ( $p^{2}$ + $q^{2}$ ).
2.	Calculate Gini for split using weighted Gini score of each node of that split.

In case of a discrete-valued attribute, the subset that gives the minimum gini index for that chosen is selected as a splitting attribute. In the case of continuous-valued attributes, the strategy is to select each pair of adjacent values as a possible split-point and point with smaller gini index chosen as the splitting point. The attribute with minimum Gini index is chosen as the splitting attribute.

# **6. Overfitting in Decision Tree algorithm** <a class="anchor" id="6"></a>
[Table of Contents](#0.1)

**Overfitting** is a practical problem while building a Decision-Tree model. The problem of overfitting is considered when the algorithm continues to go deeper and deeper to reduce the training-set error but results with an increased test-set error. So, accuracy of prediction for our model goes down. It generally happens when we build many branches due to outliers and irregularities in data.

Two approaches which can be used to avoid overfitting are as follows:-

- Pre-Pruning
- Post-Pruning

## **1. Pre-Pruning**
- In pre-pruning, we stop the tree construction a bit early. We prefer not to split a node if its goodness measure is below a threshold value. But it is difficult to choose an appropriate stopping point.

## **2. Post-Pruning**
- In post-pruning, we go deeper and deeper in the tree to build a complete tree. If the tree shows the overfitting problem then pruning is done as a post-pruning step. We use the cross-validation data to check the effect of our pruning. Using cross-validation data, we test whether expanding a node will result in improve or not. If it shows an improvement, then we can continue by expanding that node. But if it shows a reduction in accuracy then it should not be expanded. So, the node should be converted to a leaf node.

# **7. Import libraries** <a class="anchor" id="7"></a>
[Table of Contents](#0.1)

# **8. Import dataset** <a class="anchor" id="8"></a>
[Table of Contents](#0.1)

We are using the car price evaluation dataset. Link: https://www.kaggle.com/datasets/aayushsaxena0811/car-evaluationcsv

# **9. Exploratory data analysis** <a class="anchor" id="9"></a>
[Table of Contents](#0.1)

Now, I will explore the data to gain insights about the data. 
- There are 7 variables in the dataset. All the variables are of categorical data type.
- These are given by `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety` and `class`.
- `class` is the target variable.

# **10. Declare feature vector and target variable** <a class="anchor" id="10"></a>
[Table of Contents](#0.1)

    X = data.drop(['class'], axis=1)
    y = data['class']

# **11. Split data into separate training and test set** <a class="anchor" id="11"></a>
[Table of Contents](#0.1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# **12. Feature Engineering** <a class="anchor" id="12"></a>
[Table of Contents](#0.1)

- **Feature Engineering** is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power. I will carry out feature engineering on different types of variables.
- First, I will check the data types of variables again.
- Now, I will encode the categorical variables. Using category_encoders.OrdinalEncoder

# **13. Decision Tree Classifier with criterion gini index** <a class="anchor" id="13"></a>

[Table of Contents](#0.1)

    from sklearn.tree import DecisionTreeClassifier
    clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
    clf_gini.fit(X_train, y_train)
    y_pred_gini = clf_gini.predict(X_test)

- Here, **y_test** are the true class labels and **y_pred_gini** are the predicted class labels in the test-set.
- `clf_gini` is the object of DecisionTreeClassifier 

# **14. Decision Tree Classifier with criterion entropy** <a class="anchor" id="14"></a>

[Table of Contents](#0.1)

    from sklearn.tree import DecisionTreeClassifier
    clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    clf_en.fit(X_train, y_train)
    y_pred_en = clf_en.predict(X_test)

# **15. Confusion matrix** <a class="anchor" id="15"></a>
[Table of Contents](#0.1)

A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.

Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-

- **True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.

- **True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.

- **False Positives (FP)** – False Positives occur when we predict an observation belongs to a    certain class but the observation actually does not belong to that class. This type of error is called **Type I error.**

- **False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error.**

These four outcomes are summarized in a confusion matrix given below.

# **16. Classification Report** <a class="anchor" id="16"></a>
[Table of Contents](#0.1)

**Classification report** is another way to evaluate the classification model performance. It displays the  **precision**, **recall**, **f1** and **support** scores for the model. I have described these terms in later.

# **17. Results and conclusion** <a class="anchor" id="17"></a>
[Table of Contents](#0.1)

1.	In this project, I build a Decision-Tree Classifier model to predict the safety of the car. I build two models, one with criterion `gini index` and another one with criterion `entropy`. The model yields a very good performance as indicated by the model accuracy in both the cases which was found to be 0.8021.
2.	In the model with criterion `gini index`, the training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting.
3.	Similarly, in the model with criterion `entropy`, the training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021.We get the same values as in the case with criterion `gini`. So, there is no sign of overfitting.
4.	In both the cases, the training-set and test-set accuracy score is the same. It may happen because of small dataset.
5.	The confusion matrix and classification report yields very good model performance.

---

So, now we will come to the end of this repo. I hope you find this kernel useful and enjoyable. Your comments and feedback are most welcome.

Thank you

[Go to Top](#0)
