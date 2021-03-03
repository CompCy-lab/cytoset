rm(list=ls())
library(stringr)
library(dplyr)
library(CytoDx)
library(ROCR)
library(ggplot2)

##### prepare meta data #####
setwd("~/Documents/proj/CytoSet/Data/HEUvsUE/")

fcs_info_train = read.csv("train/train_labels.csv", check.names = F)
fcs_info_test = read.csv("test/test_labels.csv", check.names = F)

set.seed(12345)
ncell = 4096

##### train model #####
train_data <- fcs2DF(fcsFiles=paste(c('train/'), fcs_info_train$fcs_file, sep=''),
                     y=fcs_info_train$label,
                     assay="CyTOF",
                     fileSampleSize = ncell,
                     b=1/5,
                     excludeTransformParameters = c("Time"))

train_data = select(train_data, -Time)

x_train <- model.matrix(~., train_data[, 1:10])

fit <- CytoDx.fit(x=x_train,
                  y=train_data$y,
                  xSample=train_data$xSample,
                  family = "binomial",
                  reg = TRUE, nfold = 5, parallelCore = 1)

##### test performance #####
test_data <- fcs2DF(fcsFiles=paste(c('test/'), fcs_info_test$fcs_file, sep=''),
                    y=NULL,
                    fileSampleSize = ncell,
                    assay="CyTOF",
                    b=1/5,
                    excludeTransformParameters=c("Time"))

test_data = select(test_data, -Time)
x_test <- model.matrix(~., test_data[, 1:10])

pred <- CytoDx.pred(fit, xNew=x_test, xSampleNew=test_data$xSample)

##### plot ROC #####
t1 = prediction(predictions = pred$xNew.Pred.sample$y.Pred.1,
                labels = fcs_info_test$label)

auc = performance(t1,"auc")@y.values%>%as.numeric()

n_correct = 0
n_sample = length(t1@predictions[[1]])

for (i in 1:n_sample) {
  if (t1@predictions[[1]][i] >= 0.5 && t1@labels[[1]][i] == 1) {
    n_correct = n_correct + 1
  }
  if (t1@predictions[[1]][i] < 0.5 && t1@labels[[1]][i] == 0) {
    n_correct = n_correct + 1
  }
}
acc = n_correct / n_sample

cat("ACC for predicting HEUvsUE = ", acc, "\n")
cat("AUC for predicting HEUvsUE = ", auc, "\n")
