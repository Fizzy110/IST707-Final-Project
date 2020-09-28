dev.off() 
# Clear the console
cat('\014')  
# Clear all user objects from the environment
rm(list=ls()) 
#Load the package for my final project
library(dplyr)
library(rpart)
library(rpart.plot)
library(e1071)
library(kernlab)
library(caret)
library(class)
library(gmodels)
library(party)
library(randomForest)
library(cluster)    
library(factoextra) 
library(stats)
#Import the Final project dataset
htl <- 'http://archive.ics.uci.edu/ml/machine-learning-databases/'
parm <- 'breast-cancer-wisconsin/breast-cancer-wisconsin.data'

#Connect the URL with parm
url <- paste(htl,parm,sep = '')

#Read the data set as table format
breast <- read.table(url,sep = ',',header = FALSE,na.strings = '?')

#Rename columns based on the data set description
names(breast) <- c('ID','clumpThickness','sizeUniformity','shapeUniformity','maginalAdhesion',
                   'singleEptheliacellsize','bareNuclei','blandChromatin','normalNucleoli','mitosis','class')

#Delete the ID column
df <- breast[,-1] 

#Create the Class variable in the data set: #benign, malignant
df$class <- factor(df$class,levels = c(2,4),
                   labels = c('benign','malignant'))  
#Inspect the  final project data set
str(df)
summary(df)

#Note: for the project consistency, I will deal the NAs by removing them from the data set
#for each algorithm.

#Create train set and test set, train set size by 80%
set.seed(123)
trainset <- sample(nrow(df),0.8*nrow(df))#set the train set size as 80%
df.trainset <-df[trainset,]#training set
df.testset <- df[-trainset,]#testing set

#Inspect the frequency of data set class
table(df.trainset$class) #train set
table(df.testset$class) #test set

#Remove the NAs from test set
NAlist <- which(rowSums(is.na(df.testset)) >0)
NAs <- df.testset[NAlist,] #Remove all missing values

#Build decision tree model
set.seed(123)
#Build my decision tree model with gini
dtree <- rpart(class~., data = df.trainset, method = 'class', parms = list(split = 'gini'))

#Inspect the xerror rate
dtree$cptable
#Plot the X-val relative error and cp relationship
plotcp(dtree)
#Based on the plot, I decided use 0.01 as cp value

#Prune the decision tree with the cp value
dtree_pruned <- prune(dtree, cp = 0.01)#Based on the cp to control the tree complexity
prp (dtree_pruned, faclen = 0, 
     cex = 0.8, 
     extra = 1,
     main="Decision Tree by cp")
#Create a confusion matrix based on the tree model
dtree_pred <- predict(dtree_pruned,df.testset,type = 'class')
confusion_matrix1 <- table(df.testset$class,dtree_pred,dnn=c('Actual','Predicted'))

#Inspect the confusion matrix
confusion_matrix1
accuracy_rate1 = (96+38)/(96+2+4+38)
accuracy_rate1
#Sensitivity (True Positive Rate = TP / (TP + FN))
Sensitivity = 96/(96+2)
Sensitivity
#Specificity (True Negative Rate = TN / (TN + FP))
Specificity = 38/(38+4)
Specificity
# Precision (Positive Predictive Value = TP / (TP + FP))
Precision = 96/(96+38)
Precision
#Recall (Measure of how complete the results are = TP / (TP + FN))
Recall = 96/(96+2)
Recall
# A search engine with a high recall returns a large number of documents pertinent to the search
# F-Measure
F_measure <- (2*Precision*Recall)/(Precision + Recall)
F_measure

#Cross validation
#Before I build the cross-validation model, I should remove all the NAs first, which will 
#caused error if not been removed.
df1 <- df %>% 
  na.omit()
# Repeated K-fold CV
ctrl <- caret::trainControl(method = "repeatedcv",
                            number = 10,
                            selectionFunction = "oneSE",
                            repeats = 10)

grid <- expand.grid(model = "tree",
                    trials = c(1,5,10,15,20,25,30,35),
                    winnow = F)

models <- caret::train( class~ .,
                        data = df1,
                        method = "C5.0",
                        metric = "Kappa",
                        trControl = ctrl,
                        tuneGrid = grid)
models

#Conditional Inference Tree
#Build a conditional inference tree, and Ctree doesn not need to be pruned.
fit.ctree <- ctree(class~., data =df.trainset)
plot(fit.ctree, main = "Conditional Inference Tree")
ctree.pred <- predict(fit.ctree, df.testset, type = "response")
ctree.pref <- table(df.testset$class, ctree.pred, dnn = c("Actual","Predicted"))
ctree.pref

#Random Forest 
#Build Random Forest model
df.trainset1 <- na.omit(df.trainset)
set.seed(123)
fit.forest <- randomForest(class~., data = df.trainset1, importance = TRUE)
fit.forest
plot(fit.forest, main = "Random Forest")
#Check the importance of each variable
importance(fit.forest, type = 2)
#Perform the forest model on the testing set
forest.pred <- predict(fit.forest, df.testset)
forest.pref <- table(df.testset$class, forest.pred,
                     dnn = c("Actual","Predict"))
forest.pref

#SVM models
#First create a copy of the df for SMV model normalizing
svmdf <- df
#Reason for scaling the data: The notion is that the data are on different scales, 
# and this happenstance of how things were measured might not be desirable
#Train set size by 80%
set.seed(123)
svmtrainset <- sample(nrow(svmdf),0.80*nrow(svmdf))#set the train set size as 80%
svm.trainset <- svmdf[svmtrainset,]#training set
svm.testset <- svmdf[-svmtrainset,]#testing set
#Check the NAs distribution 
summary(svm.trainset)
summary(svm.testset)
#Remove the NAs from training and test set
svm.trainset <- na.omit(svm.trainset)
svm.testset <- na.omit(svm.testset)
#Build a tune.svm model to automatically find the best C and gamma values
set.seed(123)
tuned <- tune.svm(class~., data = svm.trainset, gamma = 10^(-6:1), cost = 10^(-10:10))
tuned
#Build a fit.svm model to find the bests kernl trick
set.seed(123)
# the idea behind scaling: after scaling, the highest value will be 1 and lowest value will be 0.
fit.svm <- svm(class~., data = svm.trainset, scale = TRUE, gamma = 0.01, cost = 1)
fit.svm
#Apply the SVM model on predicting the train and test set
pred_train <- predict(fit.svm, svm.trainset)
svm.train.conf.matrix <- table(svm.trainset$class, pred_train, dnn = c('Actual', 'Predicted'))
svm.train.conf.matrix
# Create an accuracy function
# This function divides the correct predictions (diagonal of conf. matrix) by total number of predictions
accuracy <- function(x) {
  sum( diag(x) / (sum(rowSums(x))) )*100}


accuracy(svm.train.conf.matrix)
#Predict the test set
pred_test <- predict(fit.svm, svm.testset)
svm.test.conf.matrix <- table(svm.testset$class, pred_test, dnn = c('Actual', 'Predicted'))
svm.test.conf.matrix
#Check the accuracy rate
accuracy(svm.test.conf.matrix)

#KNN
#KNN models with two type of normalizaitons Norm_min_max and Z-score.
KNNdata <- df
#Remove the missing values at the begining 
KNNdata <- na.omit(KNNdata)
# Creating a min-max normalization function
#Split into train and test set, train set size by 80%
set.seed(123)
trainset <- sample(nrow(KNNdata), 0.80* nrow(KNNdata))#set the train set size as 80%
KNN.trainset <- KNNdata[trainset,]#training set
KNN.testset <- KNNdata[-trainset,]#testing set
#Min-max normalization: the idea behind the function is the highest value will be 1 and lowest value will be 0.
norm_min_max <-function(x) { 
  ( x - min(x) ) / ( max(x) - min(x) )
}
#Apply the normalizaiton function for both train and test set
KNN.trainset1 <- KNN.trainset %>%
  mutate(sizeUniformity = norm_min_max(KNN.trainset$sizeUniformity),
         shapeUniformity = norm_min_max(KNN.trainset$shapeUniformity),
         maginalAdhesion = norm_min_max(KNN.trainset$maginalAdhesion),
         singleEptheliacellsize = norm_min_max(KNN.trainset$singleEptheliacellsize),
         bareNuclei = norm_min_max(KNN.trainset$bareNuclei),
         blandChromatin = norm_min_max(KNN.trainset$blandChromatin),
         normalNucleoli = norm_min_max(KNN.trainset$normalNucleoli),
         mitosis = norm_min_max(KNN.trainset$mitosis),
         clumpThickness = norm_min_max(KNN.trainset$clumpThickness))
KNN.testset1 <- KNN.testset %>%
  mutate(sizeUniformity = norm_min_max(KNN.testset$sizeUniformity),
         shapeUniformity = norm_min_max(KNN.testset$shapeUniformity),
         maginalAdhesion = norm_min_max(KNN.testset$maginalAdhesion),
         singleEptheliacellsize = norm_min_max(KNN.testset$singleEptheliacellsize),
         bareNuclei = norm_min_max(KNN.testset$bareNuclei),
         blandChromatin = norm_min_max(KNN.testset$blandChromatin),
         normalNucleoli = norm_min_max(KNN.testset$normalNucleoli),
         mitosis = norm_min_max(KNN.testset$mitosis),
         clumpThickness = norm_min_max(KNN.testset$clumpThickness))
# Extracting the labels from our test and train datasets
train_category <- KNNdata[trainset, 10]
test_category <- KNNdata[-trainset, 10]

# Setting our model parameters and creating the model
ctrl <- trainControl(method="repeatedcv",
                     repeats = 3)
knn1 <- train( class~ ., 
               data = KNN.trainset1, 
               method = "knn", 
               trControl = ctrl, 
               tuneLength = 20)
knn1

plot(knn1)
# Generating predictions on the test data        
knnPredict1 <- predict(knn1,
                       newdata = KNN.testset1 )

#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict1, 
                test_category)
#KNN model2 with Z-scored normalization to see if the new method impore the model performance.
KNN.trainset2 <- KNN.trainset %>%
  mutate(sizeUniformity = scale(KNN.trainset$sizeUniformity),
         shapeUniformity = scale(KNN.trainset$shapeUniformity),
         maginalAdhesion = scale(KNN.trainset$maginalAdhesion),
         singleEptheliacellsize = scale(KNN.trainset$singleEptheliacellsize),
         bareNuclei = scale(KNN.trainset$bareNuclei),
         blandChromatin = scale(KNN.trainset$blandChromatin),
         normalNucleoli = scale(KNN.trainset$normalNucleoli),
         mitosis = scale(KNN.trainset$mitosis),
         clumpThickness = scale(KNN.trainset$clumpThickness))
KNN.testset2 <- KNN.testset %>%
  mutate(sizeUniformity = scale(KNN.testset$sizeUniformity),
         shapeUniformity = scale(KNN.testset$shapeUniformity),
         maginalAdhesion = scale(KNN.testset$maginalAdhesion),
         singleEptheliacellsize = scale(KNN.testset$singleEptheliacellsize),
         bareNuclei = scale(KNN.testset$bareNuclei),
         blandChromatin = scale(KNN.testset$blandChromatin),
         normalNucleoli = scale(KNN.testset$normalNucleoli),
         mitosis = scale(KNN.testset$mitosis),
         clumpThickness = scale(KNN.testset$clumpThickness))
# Setting our model parameters and creating the model
knn2 <- train( class~ ., 
               data = KNN.trainset2, 
               method = "knn", 
               trControl = ctrl, 
               tuneLength = 20)
knn2

plot(knn2)
# Generating predictions on the test data        
knnPredict2 <- predict(knn2,
                       newdata = KNN.testset2)

#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict2, 
                test_category)
#KNN model with higher value basedm on the Z-scored normalization
#knnPredict3 <- knn(train = KNN.trainset2, test = KNN.testset2, cl = train_category , k = 10)
#CrossTable(x = test_category, y= knnPredict3, prop.chisq = FALSE)
#The higher K-value returned error, I tried different value, still could not fix it.

#Naive Bayes
#Using Naive Bayes model to predict the breast cancer
#Create a subset from the original data set and remove the NAs at the same.
NBset <- na.omit(df)
#Creating Test and Train datasets
#Train set size by 80%
set.seed(123)
NBtrain <- sample(nrow(NBset),0.80*nrow(NBset))#set the train set size as 80%
NB.trainset <-NBset[NBtrain,]#training set
NB.testset <- NBset[-NBtrain,]#testing set
#Apply the normalizaiton function for both train and test set
NB.trainset <- NB.trainset %>%
  mutate(sizeUniformity = norm_min_max(NB.trainset$sizeUniformity),
         shapeUniformity = norm_min_max(NB.trainset$shapeUniformity),
         maginalAdhesion = norm_min_max(NB.trainset$maginalAdhesion),
         singleEptheliacellsize = norm_min_max(NB.trainset$singleEptheliacellsize),
         bareNuclei = norm_min_max(NB.trainset$bareNuclei),
         blandChromatin = norm_min_max(NB.trainset$blandChromatin),
         normalNucleoli = norm_min_max(NB.trainset$normalNucleoli),
         mitosis = norm_min_max(NB.trainset$mitosis),
         clumpThickness = norm_min_max(NB.trainset$clumpThickness))
NB.testset <- NB.testset %>%
  mutate(sizeUniformity = norm_min_max(NB.testset$sizeUniformity),
         shapeUniformity = norm_min_max(NB.testset$shapeUniformity),
         maginalAdhesion = norm_min_max(NB.testset$maginalAdhesion),
         singleEptheliacellsize = norm_min_max(NB.testset$singleEptheliacellsize),
         bareNuclei = norm_min_max(NB.testset$bareNuclei),
         blandChromatin = norm_min_max(NB.testset$blandChromatin),
         normalNucleoli = norm_min_max(NB.testset$normalNucleoli),
         mitosis = norm_min_max(NB.testset$mitosis),
         clumpThickness = norm_min_max(NB.testset$clumpThickness))
# Extracting the labels from our test and train datasets
NB_train_category <- NBset[NBtrain, 10]
NB_test_category <- NBset[-NBtrain, 10]
#Build Naive Bayes model and performing predict on the testing set
NBclassifier <- naiveBayes(NB.trainset, NB_train_category)

NBpred <- predict(NBclassifier, NB.testset)
#Examine the accuracy rate
CrossTable(NBpred, NB_test_category,
           prop.chisq = F, prop.c = F, prop.r = F,
           dnn = c('predicted','actual'))
confusionMatrix(NBpred, 
                NB_test_category)
# Adjusting the model with Laplace
NBclassifier2 <- naiveBayes(NB.trainset, NB_train_category, laplace = 1)

NBpred2 <- predict(NBclassifier2, NB.testset)

CrossTable(NBpred2, NB_test_category,
           prop.chisq = F, prop.c = F, prop.r = F,
           dnn = c('predicted','actual'))
confusionMatrix(NBpred2, 
                NB_test_category)
#Clustering
#This is a test.
## redefine the data set for clustering
breast_variables <- df[,c(1,2,3,4,5,6,7,8,9)]
breast_classes <- df[,"class"]

# Standardize data and omit NAs
breast_variables<- breast_variables %>%
  na.omit() %>%
  scale()

# Check elbow plot for number of clusters
elbow_plot <- breast_variables %>% 
  fviz_nbclust(kmeans, 
               method = 'wss')

elbow_plot

# Run K-means algorithm
km_breast <- breast_variables %>%
  kmeans(centers = 2, 
         nstart = 25)

# Visualize results
fviz_cluster(km_breast, breast_variables)

