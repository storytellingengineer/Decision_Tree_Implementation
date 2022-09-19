# Decision_Tree_Implementation

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
    - 5.2 [Gini index](#5.2)
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
