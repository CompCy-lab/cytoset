library(stringr)
library(dplyr)
library(CytoDx)
library(ROCR)
library(ggplot2)
##### prepare meta data #####
setwd("~/Documents/proj/cytoset/")

fcs_info = read.csv("Data/CMV_fcs/fcs_info.csv", check.names = F)

set.seed(333)
s1 = sample(nrow(fcs_info),50)

fcs_info_train = fcs_info[s1,]
fcs_info_test = fcs_info[-s1,]

##### train model #####
train_data <- fcs2DF(fcsFiles=fcs_info_train$new_name,
                     y=fcs_info_train$CMV_Status,
                     assay="CyTOF",
                     b=1/5,
                     fileSampleSize = 5000,
                     excludeTransformParameters=c("Cell_Length"))
train_data = select(train_data,-`(Ba138)Dd`,-Bead)

x_train <- model.matrix(~.,train_data[,2:38])

fit <- CytoDx.fit(x=x_train,
                  y=train_data$y,
                  xSample=train_data$xSample,
                  family = "binomial",
                  reg = TRUE, nfold = 5, parallelCore = 5)

##### test performance #####
test_data <- fcs2DF(fcsFiles=fcs_info_test$new_name,
                    y=NULL,
                    fileSampleSize = 5000,
                    assay="CyTOF",
                    b=1/5,
                    excludeTransformParameters=c("Cell_length"))
test_data = select(test_data,-`(Ba138)Dd`,-Bead)

x_test <- model.matrix(~.,test_data[,2:38])

pred <- CytoDx.pred(fit,xNew=x_test,xSampleNew=test_data$xSample)


##### plot ROC #####
t1 = prediction(predictions = pred$xNew.Pred.sample$y.Pred.1, 
                labels = fcs_info_test$CMV_Status)
auc = performance(t1,"auc")@y.values%>%as.numeric()
cat("AUC for predicting CMV = ",auc,"\n")
roc= performance(t1, measure = "tpr", x.measure = "fpr")