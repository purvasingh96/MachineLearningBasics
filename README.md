# MachineLearningBasics
Following along with Sentdex’s Machine Learning tutorial

This follows along with the tutorial: [Scikit-learn Machine Learning with Python and SKlearn](https://www.youtube.com/playlist?list=PLQVvvaa0QuDd0flgGphKCej-9jp-QdzZ3).

## Calculating the best-fit slope and y-intercept
### The slope, m, of the best-fit line is defined as:
<img src="images/best_fit%20slope.PNG" width="100" >

### The y-intercept of the best-fit line is defined as:
<img src="images/best_fit_intercept.PNG" width="100" >

## Calculating the Coefficient of Determination
### SE - squared error of the regression line.
### r^2 - determines how "fit" the best-fit line actually is.
<img src="images/Coeff_of_determination.PNG" width="100" >

## K-Means Alogorithm
### Euclidean Distance
<img src="images//Euclidean_ditance.PNG" width="150" >

## Support Vector Machines
### Equation of a hyper-plane:
<img src="images/hyperplane_eq.PNG" width="100" >

### Equation of a hyper-plane for +ve class support vector:
<img src="images/svm_pos_class.PNG" width="130" >

### Equation of a hyper-plane for +ve class support vector:
<img src="images/svm_neg_class.PNG" width="130" >

### Equation of a decision boudry:
<img src="images/decision_boundry.PNG" width="130" >

### Equation of classification of feature set:
<img src="images/classifiying_feature_set.PNG" width="150" >

### Main optimization objective:
* **Minimize || vector(w) ||** 
* **Maximize b**

### Convex Optimiation Problem:
#### key-points:
1. The SVM's optimization problem is a convex problem, where the convex shape is the magnitude of vector w. 
2. Main Objective: find min. ||vector(w)||.
3. This point in convex graph is called **Global Minimum**. 
4. In case of non-linear optimization problems, while taking steps downwards, you might detect you are going back up, so you go back and settle in to a **Local Minimum**.

## Kernels 
1. Kernels are similarity functions that take 2 inputs and return similarity using their **dot** product. 
2. Kernels can be used to perform calculations on non-linear data involving multiple dimensions.

### Hard and Soft Margin Classifiers
1. Hard margin classifiers don't allow any slack/errors.
2. Soft margin classifiers allow us to have some **Slack (𝝃)** or exceptions in the optimization process.
3. 𝝃>=0
4. 𝝃 ∝ soft-margin
5. We want to minimize 𝝃 as much as possible, therefore, we need to minimze the equation :



     `1/2||w||^2 + (C * The sum of all of the slacks)`
     
 6. Constant **C** determines how much we want to slack to affect rest of the equation.
 
 ## Clustering and Un-supervised Machine Learning
 ### Flat Clustering 
 With flat clustering, we explicitly mention the number of clusters to the machine. E.g - **K-Means Algorithm**

### Hierarchical clustering
With Hierarchical clustering, machine itself tries to figure out, how many clusters there should be. E.g - **Mean Shift Algorithm** 






