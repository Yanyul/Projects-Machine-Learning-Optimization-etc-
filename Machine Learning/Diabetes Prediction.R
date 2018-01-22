#Group Project

rm(list = ls())

installIfAbsentAndLoad <- function(neededVector) {
    for(thispackage in neededVector) {
        if( ! require(thispackage, character.only = T) )
        { install.packages(thispackage)}
        require(thispackage, character.only = T)
    }
}

needed  <-  c("party", "glmnet", "bestglm", "MASS","pROC","class","mlbench", "modelr","gtools")  
installIfAbsentAndLoad(needed)

#load data
data(PimaIndiansDiabetes)
DiabetesOrigin <- PimaIndiansDiabetes

#Detect NA value:
sum(is.na(DiabetesOrigin)) #No missing value

#Create Response Variable
Diagnosis <- DiabetesOrigin$diabetes

#Scale down the data
DiabetesOrigin <- data.frame(scale(DiabetesOrigin[,c(-9)], center = TRUE, scale = TRUE), Diagnosis)


##### Step1: Define important variables ######################################################################

# Method 1: PCA
# Can apply to different models

pca <- prcomp(DiabetesOrigin[-9]) #already scaled down
a <- summary(pca)

b<- 1:length(DiabetesOrigin[-9])

ImportantPCA <- a$importance[2,] > summary(a$importance[2,])[2] #take PC that is greater than Median

x <- as.matrix(DiabetesOrigin[-9])

z <- x%*%a$rotation[,b[ImportantPCA]]

#This is just a simple subset selection method
#Ideal way is to do cross validation for different models and select the subset of PC

#use as cross validation criteria

########################################################################################
#Try to use plot to see what combinations of principal components explain the variance##
########################################################################################

pr.out=prcomp(DiabetesOrigin[-9], scale=TRUE)

pr.out$rotation

dim(pr.out$x)

#biplot(pr.out, scale=0)

pr.var=pr.out$sdev^2
pr.var

pve=pr.var/sum(pr.var)
pve

plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')

#Stick to what we choosed

z <- data.frame(z, Diagnosis)

#Method 2(Only Applicable for logistics models)

#Shrinkage method - deleting the close-to-zero coefficients

#reset the model data

x= model.matrix(Diagnosis~ ., DiabetesOrigin)[,-1]
y= DiabetesOrigin$Diagnosis

lasso.mod = glmnet(x,y, alpha =1, family = "binomial")
plot(lasso.mod)

set.seed(5072)
#cross validation

cv.out =cv.glmnet (x,y, alpha =1, family = "binomial")
plot(cv.out)
bestlam =cv.out$lambda.min

#Can't apply lasso to directly predict classification problem? Only useful to determine whether variables are relevant

lasso.coef<- predict(cv.out, type ="coefficients",s= bestlam)[2:length(DiabetesOrigin), ]
lasso.coef
lasso_best_subset_relevant_var <- lasso.coef != 0
lasso_best_subset_relevant_var


##### Step2: Apply possible models######################################################################
##### Step3: Use Lasso, PCA or the whole mode ##########################################################
##### Step4: Resampling applied models##################################################################


######################
#Logistics Regression#s
######################

#For logistics regression we can compare using lasso or pca

#Method 1: Use Lasso as subset selection
Diabetes <- data.frame(DiabetesOrigin[lasso_best_subset_relevant_var])

###sub step1: Using default threshold 0.5

glm.fit = glm(Diagnosis~., data = Diabetes, family = binomial)
summary (glm.fit)

glm.probs = predict(glm.fit, Diabetes, type ="response")

#Show ROC based on whole population

myROC <- roc(Diabetes$Diagnosis, glm.probs)

myROC$auc

plot(myROC, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     print.thres = TRUE, xaxs = "i", yaxs = "i")

#According to the plot, we can choose 0.356 as best threshold in this case.

###sub step: Resampling

#K-fold Cross validation(based on test error rate)
#Generate k-fold
set.seed(5072)
mydf <- Diabetes
n <- nrow(Diabetes)
mydf <- mydf[sample(1:n, n),]
num.folds <- 10
fold.indices <- cut(1:n, breaks=num.folds, labels=FALSE)

#Use Resampling method to choose between different model: in this case choose beween Logistics, LDA, QDA & KNN

test.error.rate <- 0
True_Positive <- 0

for (k in 1:num.folds){
    test.set <- mydf[which(fold.indices == k),]
    train.set <- mydf[which(fold.indices !=k), ]
    
    glm.fit = glm(Diagnosis~., data = train.set, family = "binomial")
    glm.probs = predict(glm.fit, test.set, type ="response")
    glm.pred =rep("pos",nrow(test.set))
    glm.pred[glm.probs<.5]="neg" 
    
    mytable <- table(test.set$Diagnosis,glm.pred)
    
    test.error.rate[k] <- mean((mytable[1,2] + mytable[2,1]) / sum(mytable))
    True_Positive[k] <- mytable[2,2]/sum(mytable[2, ])
}


Log_mean_lasso <- mean(test.error.rate)
Log_sd_lasso <- sd(test.error.rate)
Log_mean_TP_lasso <- mean(True_Positive)

#############################
#Method 2: Use PCA as subset selection#

Diabetes <- z

###sub step1: Using default threshold 0.5

glm.fit = glm(Diagnosis~., data = Diabetes, family = binomial)
summary (glm.fit)

glm.probs = predict(glm.fit, Diabetes, type ="response")

#Show ROC based on whole population

myROC <- roc(Diabetes$Diagnosis, glm.probs)
myROC$auc
plot(myROC, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     print.thres = TRUE, xaxs = "i", yaxs = "i")

#According to the plot, we can choose 0.328 as best threshold in this case.
#But the ROC curve seems to be good already.No point of changing it.

###sub step: Resampling

#K-fold Cross validation(based on test error rate)
#Generate k-fold
set.seed(5072)
mydf <- Diabetes
n <- nrow(Diabetes)
mydf <- mydf[sample(1:n, n),]
num.folds <- 10
fold.indices <- cut(1:n, breaks=num.folds, labels=FALSE)


#Use Resampling method to choose between different model: in this case choose beween Logistics, LDA, QDA & KNN

test.error.rate <- 0
True_Positive <- 0

for (k in 1:num.folds){
    test.set <- mydf[which(fold.indices == k),]
    train.set <- mydf[which(fold.indices !=k), ]
    
    glm.fit = glm(Diagnosis~., data = train.set, family = "binomial")
    glm.probs = predict(glm.fit, test.set, type ="response")
    glm.pred =rep("pos",nrow(test.set))
    glm.pred[glm.probs<.5]="neg" 
    
    mytable <- table(test.set$Diagnosis,glm.pred)
    
    test.error.rate[k] <- mean((mytable[1,2] + mytable[2,1]) / sum(mytable))
    True_Positive[k] <- mytable[2,2]/sum(mytable[2, ])
}

Log_mean_pca <- mean(test.error.rate)
Log_sd_pca <- sd(test.error.rate)
Log_mean_TP_pca <- mean(True_Positive)

#Method 3 Use manually subset selection method

predictors <- colnames(DiabetesOrigin[-9])
predictors
test.error.rate = 0
test.error = 0
test.error.com = 0

for (i in 2:length(predictors)){
    com <- combinations(n = length(predictors), r = i, v =predictors, repeats.allowed = FALSE)
    for (j in 1:nrow(com)){
        Diabetes <- data.frame(DiabetesOrigin[com[j, ]],Diagnosis)
        glm.fit = glm(Diagnosis~., data = Diabetes, family = binomial)
        glm.probs = predict(glm.fit, Diabetes, type ="response")
        glm.pred =rep("pos",nrow(Diabetes))
        glm.pred[glm.probs<.5]="neg" 
        
        mytable <- table(Diabetes$Diagnosis,glm.pred)
        
        test.error.rate[j] <- mean((mytable[1,2] + mytable[2,1]) / sum(mytable))
    }
    test.error[i] = min(test.error.rate)
    test.error.com[i] = which.min(test.error.rate)
}

test.error
test.error.com
Bestsubset <- combinations(n = length(predictors), r = 7, v =predictors, repeats.allowed = FALSE)[1, ]

Bestsubset

#Print out the same subset selection as lasso.

###############################
# Linear Discriminant Analysis#
###############################


#Not use subset selection
#Diabetes <- data.frame(DiabetesOrigin)

#Method 1
#Use Mannually subset selection

predictors <- colnames(DiabetesOrigin[-9])
predictors
test.error.rate = 0
test.error = 0
test.error.com = 0

for (i in 2:length(predictors)){
    com <- combinations(n = length(predictors), r = i, v =predictors, repeats.allowed = FALSE)
    for (j in 1:nrow(com)){
        Diabetes <- data.frame(DiabetesOrigin[com[j, ]],Diagnosis)
        lda.fit = lda(Diagnosis~., data = Diabetes)
        lda.pred = predict(lda.fit, Diabetes)
        
        lda.class =lda.pred$class
        
        mytable <- table(Diabetes$Diagnosis, lda.class)
        
        test.error.rate[j] <- mean((mytable[1,2] + mytable[2,1]) / sum(mytable))
    }
    test.error[i] = min(test.error.rate)
    test.error.com[i] = which.min(test.error.rate)
}

test.error
test.error.com

Bestsubset <- combinations(n = length(predictors), r = 7, v =predictors, repeats.allowed = FALSE)[1, ]

Diabetes = data.frame(DiabetesOrigin[Bestsubset], Diagnosis)

lda.fit = lda(Diagnosis~ ., data = Diabetes)
lda.fit

lda.pred = predict(lda.fit, Diabetes)

probs <- lda.pred$posterior[,2]
myROC <- roc(Diabetes$Diagnosis, probs)

myROC$auc
plot(myROC, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     print.thres = TRUE, xaxs = "i", yaxs = "i")

###sub step: Resampling

set.seed(5072)
mydf <- Diabetes
n <- nrow(Diabetes)
mydf <- mydf[sample(1:n, n),]
num.folds <- 10
fold.indices <- cut(1:n, breaks=num.folds, labels=FALSE)

test.error.rate <- 0
True_Positive <- 0

for (k in 1:num.folds){
    test.set <- mydf[which(fold.indices == k),]
    train.set <- mydf[which(fold.indices !=k), ]

    lda.fit = lda(Diagnosis~., data = train.set)
    lda.pred = predict(lda.fit, test.set)

    lda.class =lda.pred$class

    mytable <- table(test.set$Diagnosis, lda.class)

    test.error.rate[k] <- mean((mytable[1,2] + mytable[2,1]) / sum(mytable))
    True_Positive[k] <- mytable[2,2]/sum(mytable[2, ])
}

LDA_mean_sub <- mean(test.error.rate)
LDA_sd_sub <- sd(test.error.rate)
LDA_mean_TP_sub <- mean(True_Positive)


#Method 2 Use PCA as subset selection
Diabetes <- z

###sub step1: Using default threshold 0.5
lda.fit = lda(Diagnosis~ ., data = Diabetes)
lda.fit 

lda.pred = predict(lda.fit, Diabetes)

probs <- lda.pred$posterior[,2]
myROC <- roc(Diabetes$Diagnosis, probs)

myROC$auc
plot(myROC, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     print.thres = TRUE, xaxs = "i", yaxs = "i")


###sub step: Resampling 

set.seed(5072)
mydf <- Diabetes
n <- nrow(Diabetes)
mydf <- mydf[sample(1:n, n),]
num.folds <- 10
fold.indices <- cut(1:n, breaks=num.folds, labels=FALSE)

test.error.rate <- 0
True_Positive <- 0

for (k in 1:num.folds){
    test.set <- mydf[which(fold.indices == k),]
    train.set <- mydf[which(fold.indices !=k), ]
    
    lda.fit = lda(Diagnosis~., data = train.set)
    lda.pred = predict(lda.fit, test.set)
    
    lda.class =lda.pred$class
    
    mytable <- table(test.set$Diagnosis, lda.class) 
    
    test.error.rate[k] <- mean((mytable[1,2] + mytable[2,1]) / sum(mytable))
    True_Positive[k] <- mytable[2,2]/sum(mytable[2, ])
}

LDA_mean_pca <- mean(test.error.rate)
LDA_sd_pca <- sd(test.error.rate)
LDA_mean_TP_pca <- mean(True_Positive)

##################################
# Quadratic Discriminant Analysis#
##################################

#Not use subset selection
#Diabetes <- data.frame(DiabetesOrigin)

#Method 1
#Use Mannually subset selection
#Method 1
#Use Mannually subset selection

predictors <- colnames(DiabetesOrigin[-9])
predictors
test.error.rate = 0
test.error = 0
test.error.com = 0

for (i in 2:length(predictors)){
    com <- combinations(n = length(predictors), r = i, v =predictors, repeats.allowed = FALSE)
    for (j in 1:nrow(com)){
        Diabetes <- data.frame(DiabetesOrigin[com[j, ]],Diagnosis)
        qda.fit = qda(Diagnosis~., data = Diabetes)
        qda.pred = predict(qda.fit, Diabetes)
        
        qda.class =qda.pred$class
        
        mytable <- table(Diabetes$Diagnosis, qda.class)
        
        test.error.rate[j] <- mean((mytable[1,2] + mytable[2,1]) / sum(mytable))
    }
    test.error[i] = min(test.error.rate)
    test.error.com[i] = which.min(test.error.rate)
}

test.error
test.error.com

Bestsubset <- combinations(n = length(predictors), r = 4, v =predictors, repeats.allowed = FALSE)[6, ]

Diabetes = data.frame(DiabetesOrigin[Bestsubset], Diagnosis)

qda.fit = qda(Diagnosis~ ., data = Diabetes)
qda.fit

qda.pred = predict(qda.fit, Diabetes)

probs <- qda.pred$posterior[,2]
myROC <- roc(Diabetes$Diagnosis, probs)

myROC$auc
plot(myROC, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     print.thres = TRUE, xaxs = "i", yaxs = "i")

###sub step: Resampling

set.seed(5072)
mydf <- Diabetes
n <- nrow(Diabetes)
mydf <- mydf[sample(1:n, n),]
num.folds <- 10
fold.indices <- cut(1:n, breaks=num.folds, labels=FALSE)

test.error.rate <- 0
True_Positive <- 0

for (k in 1:num.folds){
    test.set <- mydf[which(fold.indices == k),]
    train.set <- mydf[which(fold.indices !=k), ]

    qda.fit = qda(Diagnosis~., data = train.set)
    qda.pred = predict(qda.fit, test.set)

    qda.class =qda.pred$class

    mytable <- table(test.set$Diagnosis, qda.class)

    test.error.rate[k] <- mean((mytable[1,2] + mytable[2,1]) / sum(mytable))
    True_Positive[k] <- mytable[2,2]/sum(mytable[2, ])
}

QDA_mean_sub <- mean(test.error.rate)
QDA_sd_sub <- sd(test.error.rate)
QDA_mean_TP_sub <- mean(True_Positive)


#Use PCA as subset selection
Diabetes <- z

###sub step1: Using default threshold 0.5
qda.fit = qda(Diagnosis~ ., data = Diabetes)
qda.fit 

qda.pred = predict(qda.fit, Diabetes)

probs <- qda.pred$posterior[,2]
myROC <- roc(Diabetes$Diagnosis, probs)

myROC$auc
plot(myROC, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     print.thres = TRUE, xaxs = "i", yaxs = "i")


###sub step: Resampling 

set.seed(5072)
mydf <- Diabetes
n <- nrow(Diabetes)
mydf <- mydf[sample(1:n, n),]
num.folds <- 10
fold.indices <- cut(1:n, breaks=num.folds, labels=FALSE)

test.error.rate <- 0
True_Positive <- 0

for (k in 1:num.folds){
    test.set <- mydf[which(fold.indices == k),]
    train.set <- mydf[which(fold.indices !=k), ]
    
    qda.fit = qda(Diagnosis~., data = train.set)
    qda.pred = predict(qda.fit, test.set)
    
    qda.class =qda.pred$class
    
    mytable <- table(test.set$Diagnosis, qda.class) 
    
    test.error.rate[k] <- mean((mytable[1,2] + mytable[2,1]) / sum(mytable))
    True_Positive[k] <- mytable[2,2]/sum(mytable[2, ])
}

QDA_mean_pca <- mean(test.error.rate)
QDA_sd_pca <- sd(test.error.rate)
QDA_mean_TP_pca <- mean(True_Positive)



###############################
#          KNN Method         #
###############################

#Not use subset selection
#DiabetesKNN <- data.frame(DiabetesOrigin)

#Method 1: Use Mannually subset selection

set.seed(5072)

mydf <- DiabetesOrigin
n <- nrow(Diabetes)
mydf <- mydf[sample(1:n, n),]
num.folds <- 10
fold.indices <- cut(1:n, breaks=num.folds, labels=FALSE)

test.error.rate <- matrix(nrow = num.folds, ncol = 50, dimnames = list(c(1:10), c(1:50)))

for (i in 1:num.folds){
    test.set <- mydf[which(fold.indices == i),]
    train.set <- mydf[which(fold.indices !=i), ]
    train.x <- train.set[-length(names(train.set))]
    train.y <- train.set$Diagnosis
    test.x <- test.set[-length(names(test.set))]
    test.y <- test.set$Diagnosis
    
    for (k in 1:50){
        knn.pred <- knn(train.x, test.x,  train.y, k = k)
        test.error.rate[i,k] <- mean(test.y != knn.pred)
    }
}
test.error.mean <- 0

for (i in 1:50){
    test.error.mean[i] <- mean(test.error.rate[ ,i])
}

which.min(test.error.mean)



library(gtools)

predictors <- colnames(DiabetesOrigin[-9])

predictors

test.error.rate = 0
test.error = 0
test.error.com = 0

for (i in 2:length(predictors)){
    com <- combinations(n = length(predictors), r = i, v =predictors, repeats.allowed = FALSE)
    for (j in 1:nrow(com)){
        Diabetes <- data.frame(DiabetesOrigin[com[j, ]],Diagnosis)
        x <- Diabetes[-length(Diabetes)]
        y <- Diabetes$Diagnosis
        
        knn.pred <- knn(x, x,  y, k = which.min(test.error.mean))
        mytable <- table(y,knn.pred) 
        #print(mytable)
        
        test.error.rate[j] <- mean(knn.pred != test.y)
    }
    test.error[i] = min(test.error.rate)
    test.error.com[i] = which.min(test.error.rate)
}

test.error
test.error.com

Bestsubset <- combinations(n = length(predictors), r = 3, v =predictors, repeats.allowed = FALSE)[46, ]

Diabetes = data.frame(DiabetesOrigin[Bestsubset], Diagnosis)

set.seed(5072)

mydf <- Diabetes
n <- nrow(Diabetes)
mydf <- mydf[sample(1:n, n),]
num.folds <- 10
fold.indices <- cut(1:n, breaks=num.folds, labels=FALSE)

test.error.rate <- matrix(nrow = num.folds, ncol = 50, dimnames = list(c(1:10), c(1:50)))

for (i in 1:num.folds){
    test.set <- mydf[which(fold.indices == i),]
    train.set <- mydf[which(fold.indices !=i), ]
    train.x <- train.set[-length(names(train.set))]
    train.y <- train.set$Diagnosis
    test.x <- test.set[-length(names(test.set))]
    test.y <- test.set$Diagnosis
    
    for (k in 1:50){
        knn.pred <- knn(train.x, test.x,  train.y, k = k)
        test.error.rate[i,k] <- mean(test.y != knn.pred)
    }
}
test.error.mean <- 0

for (i in 1:50){
    test.error.mean[i] <- mean(test.error.rate[ ,i])
}

test.error.mean
plot(test.error.mean, type = "b")

which.min(test.error.mean)
points(which.min(test.error.mean), min(test.error.mean), pch = 19, cex = 2, col = "red")


#Use best k


test.error.rate <- 0
True_Positive <- 0

for (i in 1:num.folds){
    test.set <- mydf[which(fold.indices == i),]
    train.set <- mydf[which(fold.indices !=i), ]
    train.x <- train.set[-length(names(train.set))]
    train.y <- train.set$Diagnosis
    test.x <- test.set[-length(names(test.set))]
    test.y <- test.set$Diagnosis
    
    knn.pred <- knn(train.x, test.x,  train.y, k = which.min(test.error.mean))
    mytable <- table(test.y,knn.pred) 
    #print(mytable)
    
    test.error.rate[i] <- mean(knn.pred != test.y)
    True_Positive[i] <- mytable[2,2]/sum(mytable[2, ])
}    

KNN_mean_sub <- mean(test.error.rate)
KNN_sd_sub <- sd(test.error.rate)
KNN_mean_TP_sub <- mean(True_Positive)


#Method 2: Use PCA as subset selection
Diabetes <- z

###sub step2: Resampling 

set.seed(5072)

mydf <- Diabetes
n <- nrow(Diabetes)
mydf <- mydf[sample(1:n, n),]
num.folds <- 10
fold.indices <- cut(1:n, breaks=num.folds, labels=FALSE)

test.error.rate <- matrix(nrow = num.folds, ncol = 50, dimnames = list(c(1:10), c(1:50)))

for (i in 1:num.folds){
    test.set <- mydf[which(fold.indices == i),]
    train.set <- mydf[which(fold.indices !=i), ]
    train.x <- train.set[-length(names(train.set))]
    train.y <- train.set$Diagnosis
    test.x <- test.set[-length(names(test.set))]
    test.y <- test.set$Diagnosis
    
    for (k in 1:50){
        knn.pred <- knn(train.x, test.x,  train.y, k = k)
        test.error.rate[i,k] <- mean(test.y != knn.pred)
    }
}
test.error.mean <- 0

for (i in 1:50){
    test.error.mean[i] <- mean(test.error.rate[ ,i])
}

test.error.mean
plot(test.error.mean, type = "b")

which.min(test.error.mean)
points(which.min(test.error.mean), min(test.error.mean), pch = 19, cex = 2, col = "red")


#Use best k


test.error.rate <- 0
True_Positive <- 0

for (i in 1:num.folds){
    test.set <- mydf[which(fold.indices == i),]
    train.set <- mydf[which(fold.indices !=i), ]
    train.x <- train.set[-length(names(train.set))]
    train.y <- train.set$Diagnosis
    test.x <- test.set[-length(names(test.set))]
    test.y <- test.set$Diagnosis
    
    knn.pred <- knn(train.x, test.x,  train.y, k = which.min(test.error.mean))
    mytable <- table(test.y,knn.pred) 
    #print(mytable)
    
    test.error.rate[i] <- mean(knn.pred != test.y)
    True_Positive[i] <- mytable[2,2]/sum(mytable[2, ])
}    

KNN_mean_pca <- mean(test.error.rate)
KNN_sd_pca <- sd(test.error.rate)
KNN_mean_TP_pca <- mean(True_Positive)


##### Step5: Choosing models based on CV Test error rate ##############################################################

ModelCompare <- matrix(nrow =  8, ncol = 3, dimnames = list(c("Logistics-LASSO(sub)","Logistics-PCA", 
                                                              "LDA-sub","LDA-PCA", "QDA-sub","QDA-PCA", 
                                                              "KNN-sub","KNN-PCA"), c("Mean Error Rate", "Error Rate SD", "Mean_TP")))

ModelCompare[1,1] <- Log_mean_lasso
ModelCompare[1,2] <- Log_sd_lasso 
ModelCompare[1,3] <- Log_mean_TP_lasso

ModelCompare[2,1] <- Log_mean_pca 
ModelCompare[2,2] <- Log_sd_pca  
ModelCompare[2,3] <- Log_mean_TP_pca  


ModelCompare[3,1] <- LDA_mean_sub
ModelCompare[3,2] <- LDA_sd_sub
ModelCompare[3,3] <- LDA_mean_TP_sub


ModelCompare[4,1] <- LDA_mean_pca
ModelCompare[4,2] <- LDA_sd_pca
ModelCompare[4,3] <- LDA_mean_TP_pca


ModelCompare[5,1] <- QDA_mean_sub 
ModelCompare[5,2] <- QDA_sd_sub 
ModelCompare[5,3] <- QDA_mean_TP_sub

ModelCompare[6,1] <- QDA_mean_pca 
ModelCompare[6,2] <- QDA_sd_pca
ModelCompare[6,3] <- QDA_mean_TP_pca

ModelCompare[7,1] <- KNN_mean_sub 
ModelCompare[7,2] <- KNN_sd_sub
ModelCompare[7,3] <- KNN_mean_TP_sub

ModelCompare[8,1] <- KNN_mean_pca  
ModelCompare[8,2] <- KNN_sd_pca 
ModelCompare[8,3] <- KNN_mean_TP_pca 

ModelCompare

#In this case we know that we can use Logistics-sub/lasso Model or LDA-sub model because it results in a lower error rate and lower sd.
#use all predictors because makes no difference

#use sub/lasso population

#For logistics

Diabetes <- data.frame(DiabetesOrigin[lasso_best_subset_relevant_var])

glm.fit = glm(Diagnosis~., data = Diabetes, family = "binomial")
glm.probs = predict(glm.fit, Diabetes, type ="response")
myROC <- roc(Diagnosis, glm.probs)

plot(myROC, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     print.thres = TRUE, xaxs = "i", yaxs = "i")

glm.pred =rep ("pos",nrow(Diabetes))
glm.pred [glm.probs >.356]="neg" 

mytable <- table(Diagnosis,glm.pred)
mytable

#Error rate
errorrate <- (mytable[1,2] + mytable[2,1]) / sum(mytable)
errorrate

#We care more about true positive rate, we want to maximize that
True_Positive <- mytable[2,2]/sum(mytable[2, ])
True_Positive


#use resampling
test.error.rate <- 0
True_Positive <- 0

for (k in 1:num.folds){
    test.set <- mydf[which(fold.indices == k),]
    train.set <- mydf[which(fold.indices !=k), ]
    
    glm.fit = glm(Diagnosis~., data = train.set, family = "binomial")
    glm.probs = predict(glm.fit, test.set, type ="response")
    glm.pred =rep("pos",nrow(test.set))
    glm.pred[glm.probs<.356]="neg" 
    
    mytable <- table(test.set$Diagnosis,glm.pred)
    
    test.error.rate[k] <- mean((mytable[1,2] + mytable[2,1]) / sum(mytable))
    True_Positive[k] <- mytable[2,2]/sum(mytable[2, ])
}


mean(test.error.rate)
sd(test.error.rate)
mean(True_Positive)

#Display trade off of different thres

#Display trade off of different thres

thres <- seq(0.05,0.95,0.05)

glm.fit = glm(Diagnosis~., data = Diabetes, family = "binomial")
glm.probs = predict(glm.fit, Diabetes, type ="response")
myROC <- roc(Diagnosis, glm.probs)

type1FalsePositiveErrorRate <- 0
type2FalseNegativeErrorRate <- 0

k = 1

for (i in thres){
    glm.pred =rep ("pos",nrow(Diabetes))
    glm.pred[glm.probs < i]="neg" 
    
    mytable <- table(Diabetes$Diagnosis,glm.pred)
    
    type1FalsePositiveErrorRate[k] <- mytable[1, 2] / sum(mytable[1, ])
    type2FalseNegativeErrorRate[k] <- mytable[2, 1] / sum(mytable[2, ])
    k = k + 1
}

type1FalsePositiveErrorRate1 <- 0

for (i in 1:length(type1FalsePositiveErrorRate)){
    type1FalsePositiveErrorRate1[i] <- type1FalsePositiveErrorRate[length(type1FalsePositiveErrorRate)- i + 1]
}

type2FalseNegativeErrorRate1 <- 0

for (i in 1:length(type1FalsePositiveErrorRate)){
    type2FalseNegativeErrorRate1[i] <- type2FalseNegativeErrorRate[length(type2FalseNegativeErrorRate)- i + 1]
}

thres1 <- 0

for (i in 1:length(thres)){
    thres1[i] <- thres[length(thres)- i + 1]
}

plot(x = thres1, y = type1FalsePositiveErrorRate1, xlab = "Thresholds", ylab = "Error Rate",type = "b", col = "red",xlim = c(1,0))
lines(x = thres1, y = type2FalseNegativeErrorRate1, type = "b", col = "blue",xlim = c(1,0))
legend("topright", col = c("red", "blue"), pch = 16, legend = c("Type I Error rate","Type II Error rate"))
title(main = "Type I and Type II Error rates with different thresholds")
points(x = thres1[14:19], y = type2FalseNegativeErrorRate1[14:19], col = "blue", pch = 16)
points(x = thres1[14:19], y = type1FalsePositiveErrorRate1[14:19], col = "red", pch = 16)

ThresholdCompare <- data.frame(thres, type1FalsePositiveErrorRate, type2FalseNegativeErrorRate)
ThresholdCompare

#For LDA:

lda.fit = lda(Diagnosis~ ., data = Diabetes)
lda.fit

lda.pred = predict(lda.fit, Diabetes)

probs <- lda.pred$posterior[,2]
myROC <- roc(Diabetes$Diagnosis, probs)

myROC$auc
plot(myROC, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     print.thres = TRUE, xaxs = "i", yaxs = "i")


###sub step3: Resampling - change threshold

test.error.rate <- 0
True_Positive <- 0

for (k in 1:num.folds){
    test.set <- mydf[which(fold.indices == k),]
    train.set <- mydf[which(fold.indices !=k), ]
    
    lda.fit = lda(Diagnosis~., data = train.set)
    lda.pred = predict(lda.fit, test.set)
    
    lda.pred_table <- lda.pred$posterior
    lda.pred_new <- as.numeric(lda.pred_table[,2])
    
    lda.class_new =rep("pos",length(test.set$ Diagnosis))
    lda.class_new[lda.pred_new < .302] ="neg"
    
    mytable <- table(test.set$Diagnosis, lda.class_new) 
    
    test.error.rate[k] <- mean((mytable[1,2] + mytable[2,1]) / sum(mytable))
    True_Positive[k] <- mytable[2,2]/sum(mytable[2, ])
}

mean(test.error.rate)
sd(test.error.rate)
mean(True_Positive)

#Threshold
type1FalsePositiveErrorRate <- 0
type2FalseNegativeErrorRate <- 0

k = 1

for (i in thres){
    lda.fit = lda(Diagnosis~., data = Diabetes)
    lda.pred = predict(lda.fit, Diabetes)
    
    lda.pred_table <- lda.pred$posterior
    lda.pred_new <- as.numeric(lda.pred_table[,2])
    
    lda.class_new =rep("pos",length(Diabetes$Diagnosis))
    lda.class_new[lda.pred_new < i] ="neg"
    
    mytable <- table(Diabetes$Diagnosis, lda.class_new)
    
    type1FalsePositiveErrorRate[k] <- mytable[1, 2] / sum(mytable[1, ])
    type2FalseNegativeErrorRate[k] <- mytable[2, 1] / sum(mytable[2, ])
    k = k + 1
}

type1FalsePositiveErrorRate1 <- 0

for (i in 1:length(type1FalsePositiveErrorRate)){
    type1FalsePositiveErrorRate1[i] <- type1FalsePositiveErrorRate[length(type1FalsePositiveErrorRate)- i + 1]
}

type2FalseNegativeErrorRate1 <- 0

for (i in 1:length(type1FalsePositiveErrorRate)){
    type2FalseNegativeErrorRate1[i] <- type2FalseNegativeErrorRate[length(type2FalseNegativeErrorRate)- i + 1]
}

plot(x = thres1, y = type1FalsePositiveErrorRate1, xlab = "Thresholds", ylab = "Error Rate",type = "b", col = "red", xlim=c(1,0))
lines(x = thres1, y = type2FalseNegativeErrorRate1, type = "b", col = "blue",xlim=c(1,0))
legend("topright", col = c("red", "blue"), pch = 16, legend = c("Type I Error rate","Type II Error rate"))
title(main = "Type I and Type II Error rates with different thresholds")
points(x = thres1[14:19], y = type2FalseNegativeErrorRate1[14:19], col = "blue", pch = 16)
points(x = thres1[14:19], y = type1FalsePositiveErrorRate1[14:19], col = "red", pch = 16)

ThresholdCompare <- data.frame(thres, type1FalsePositiveErrorRate1, type2FalseNegativeErrorRate1)
ThresholdCompare




#Sth fancy

library(plotly)
library(dplyr)

p <- plot_ly(data = ThresholdCompare, x = thres, y = type1FalsePositiveErrorRate, z = type2FalseNegativeErrorRate, mode = "markers") %>%
    layout(
        title = "Type I Error and Type II Error in Threshold",
        scene = list(
            xaxis = list(title = "Threshold"),
            yaxis = list(title = "Type I Error"),
            zaxis = list(title = "Type II Error")
        ))
p


