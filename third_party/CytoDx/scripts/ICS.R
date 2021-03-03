rm(list=ls())
library(stringr)
library(dplyr)
library(CytoDx)
library(ROCR)
library(ggplot2)

##### prepare meta data #####
setwd("~/Documents/proj/CytoSet/Data/ICS/")

fcs_info_train = read.csv("train/train_labels.csv", check.names = F)
fcs_info_test = read.csv("test/test_labels.csv", check.names = F)

ncell= 256
seed = 12345
set.seed(seed)


##### train model #####
train_data <- fcs2DF(fcsFiles=paste(c('train/'), fcs_info_train$fcs_filename, sep=''),
                     y=fcs_info_train$label,
                     assay="CyTOF",
                     fileSampleSize = ncell,
                     b=1/5,
                     excludeTransformParameters = c('Time', 'Cell_length'))

train_data = select(train_data, -Time)

x_train <- model.matrix(~.,train_data[,1:11])

fit <- CytoDx.fit(x=x_train,
                  y=train_data$y,
                  xSample=train_data$xSample,
                  family = "binomial",
                  reg = TRUE, nfold = 5, parallelCore = 1)

##### test performance #####
test_data <- fcs2DF(fcsFiles=paste(c('test/'), fcs_info_test$fcs_filename, sep=''),
                    y=NULL,
                    fileSampleSize = ncell,
                    assay="CyTOF",
                    b=1/5,
                    excludeTransformParameters=c("Time", "Cell_length"))

test_data = select(test_data, -Time)
x_test <- model.matrix(~., test_data[,1:11])

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


fpr = performance(t1, 'fpr')@y.values[[1]]
tpr = performance(t1, 'tpr')@y.values[[1]]

res_df = data.frame(fpr = fpr, tpr = tpr)
write.csv(res_df, paste("~/Documents/proj/CytoSet/third_party/CytoDx/exp/ICS", ncell, seed, sep="_"))

cat("ACC for predicting NK_cell = ", acc, "\n")
cat("AUC for predicting NK_cell = ", auc, "\n")
