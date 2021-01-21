library(dplyr)
library(tidyr)
library(CytoDx)
library(ROCR)
library(ggplot2)
library(rpart)
library(flowCore)

##### before start #####
# 1. This code predict HAI titer using CytoDx. The code reproduces
# Fig. 4-5.

# 2. Set the working directory to the "Source_Code" folder.

# 3. The data used in this code are downloaded from SDY404 and SDY112 of ImmPort

##### load data #####
# load vaccine response data
hai_result=read.csv("Data/hai_data.csv",stringsAsFactors = F)

# transform and summarize data
hai_result = hai_result%>%
  spread(key=STUDY_TIME_COLLECTED,value = VALUE_REPORTED)%>%
  mutate("0"=log2(`0`),"28"=log2(`28`))%>%
  mutate("Diff"=`28`-`0`)%>%
  select(-VIRUS_STRAIN_REPORTED)%>%
  group_by(SUBJECT_ACCESSION,STUDY_ACCESSION )%>%
  summarise_all(.funs=mean)%>%as.data.frame()

# load cytometry meta-data
fcs_info = read.csv("Data/HAI_fcs/fcs_info.csv",stringsAsFactors = F)
fcs_info = mutate(fcs_info,"new_name"=paste0("Data/HAI_fcs/",new_name))

# join cytometry and vaccine data
fcs_info = inner_join(fcs_info,hai_result,
                      by=c("SUBJECT_ACCESSION","STUDY_ACCESSION"))

# load marker dictionary for name standardization
load("Data/HAI_fcs/nameDict.rda")

##### filter and plot #####

# filter age group
fcs_info = fcs_info%>%dplyr::filter(Subject.Age<=35| Subject.Age>=60)

# filter cytometry panels
fcs_info = fcs_info%>%select(-Gender,-Race,-Biosample.Type)%>%na.omit()%>%
  mutate("type"=gsub("^.*_|-.*$","",panel_id))


# plot age 
p=ggplot(data=fcs_info,aes(x=STUDY_ACCESSION, y=Subject.Age,color =type))+
  geom_jitter(width = 0.2,size=2)+theme_bw()
plot(p)

# save the fcs_info for all data
fcs_info_all = fcs_info


##### transform for young #####
fcs_info=fcs_info_all%>%dplyr::filter(Subject.Age<=35)

# rank transform the vaccine response
fcs_info = fcs_info%>%group_by(STUDY_ACCESSION)%>%
  mutate("0_rank"=rank.ub.average(`0`))%>%
  mutate("Diff_rank"=rank.ub.average(Diff))%>%
  mutate("28_rank"=rank.ub.average(`28`))%>%
  as.data.frame()

p=ggplot(data=fcs_info,aes(x=STUDY_ACCESSION, y=`28`,color =`28_rank`<=50))+
  geom_jitter(width = 0.2,size=2)+theme_bw()+ylim(2,9)
plot(p)

##### build predictive model for young #####

# read train data 
fcs_info_train = fcs_info%>%
  dplyr::filter(panel_id=="Data_SDY112_CyTOF-1")%>%
  na.omit()

train_data = fcs2DF(fcsFiles=fcs_info_train$new_name,
                    y=(fcs_info_train$`28_rank`>50),
                    assay="CyTOF",
                    b=1/5,
                    fileSampleSize=5000,
                    compFiles=NULL,
                    nameDict=nameDict,
                    excludeTransformParameters=
                      c("FSC-A","FSC-W","FSC-H","Time","Cell_length"))

# parepare data  
common_marker=c("CD4","CCR7","CD3","CD45RA","HLADR")
x_train = pRank(x=train_data[,common_marker],
                xSample=train_data$xSample)
x_train = model.matrix(~.*.,x_train)

# train CytoDx
fit = CytoDx.fit(x=x_train,y=train_data$y,xSample=train_data$xSample,
                 family = "binomial",reg = F)



##### predict in SDY404 FCM for young #####

# read SDY404 cytometry data
fcs_info_test = fcs_info%>%
  dplyr::filter(panel_id=="Data_SDY404_FCM-8")%>%
  na.omit()
test_data = fcs2DF(fcsFiles=fcs_info_test$new_name,
                   y=fcs_info_test$`28_rank`>50,
                   assay="FCM",
                   b=1/150,
                   fileSampleSize=5000,
                   compFiles=NULL,
                   nameDict=nameDict,
                   excludeTransformParameters=
                     c("FSC-A","FSC-W","FSC-H","Time","Cell_length"))


# predict using CytoDx
x_test= pRank(x=test_data[,common_marker],
              xSample=test_data$xSample)

x_test = model.matrix(~.*.,x_test)
pred = CytoDx.pred(fit,xNew=x_test,xSampleNew = test_data$xSample)

# evaluate result
y_test = test_data[,c("xSample","y")]%>%
  group_by(xSample)%>%summarise("y"=unique(y))%>%
  select(-xSample)%>%unlist()
t1 = prediction(predictions = pred$xNew.Pred.sample$y.Pred.s0, 
                labels = y_test)
auc = performance(t1,"auc")@y.values%>%as.numeric()
cat("AUC for young individual = ",auc,"\n")
# plot result
roc1= performance(t1, measure = "tpr", x.measure = "fpr")
save(test_data,train_data,common_marker,file = "Data/young.rda")

##### plot 2D #####

# plot training
data_T = train_data
p = ggplot(data_T, aes(x=CD3, y=CD4)) + 
  stat_binhex(bins=60)+
  scale_fill_distiller(palette = "RdBu",limits=c(0,500))+
  theme_bw()
plot(p)

data_T = pRank(data_T[,common_marker],xSample = data_T$xSample)
p = ggplot(data_T, aes(x=CD3, y=CD4)) + 
  stat_binhex(bins=60)+
  scale_fill_distiller(palette = "RdBu",limits=c(0,150))+
  theme_bw()
plot(p)
# plot testing data
data_T = test_data%>%dplyr::filter(CD3<4.5&CD3>-1&CD4<6&CD4>-3)
p = ggplot(data_T, aes(x=CD3, y=CD4)) + 
  stat_binhex(bins=60)+
  scale_fill_distiller(palette = "RdBu")+
  theme_bw()
plot(p)

data_T = pRank(data_T[,common_marker],xSample = data_T$xSample)
p = ggplot(data_T, aes(x=CD3, y=CD4)) + 
  stat_binhex(bins=60)+
  scale_fill_distiller(palette = "RdBu",limits=c(0,100))+
  theme_bw()
plot(p)

##### plot population #####
TG = treeGate(fit$train.Data.cell$y.Pred.s0,
              x=train_data[,common_marker],
              control=rpart.control(maxdepth=3))


sub_data_2 = test_data%>%dplyr::filter(CCR7>1&HLADR<1&CD45RA>1&CD3>1.7)%>%
  group_by(xSample)%>%summarise(n=n()/5000*100,y=unique(y))%>%
  mutate("SDY"="SDY404")
t1 = t.test(sub_data_2$n~sub_data_2$y)$p.value
p1=ggplot(sub_data_2,aes(x=y,y=n,color=y))+
  geom_boxplot(size=1)+
  theme_bw()+
  ggtitle(t1)
plot(p1)

sub_data_2 = test_data%>%dplyr::filter(CCR7>1&HLADR<1&CD45RA>1&CD3>1.7&CD4>3)%>%
  group_by(xSample)%>%summarise(n=n()/5000*100,y=unique(y))%>%
  mutate("SDY"="SDY404")
t1 = t.test(sub_data_2$n~sub_data_2$y)$p.value
p2=ggplot(sub_data_2,aes(x=y,y=n,color=y))+
  geom_boxplot(size=1)+
  theme_bw()+
  ggtitle(t1)
plot(p2)

sub_data_2 = test_data%>%dplyr::filter(CCR7>1&HLADR<1&CD45RA>1&CD3>1.7&CD4<3)%>%
  group_by(xSample)%>%summarise(n=n()/5000*100,y=unique(y))%>%
  mutate("SDY"="SDY404")
t1 = t.test(sub_data_2$n~sub_data_2$y)$p.value
p3=ggplot(sub_data_2,aes(x=y,y=n,color=y))+
  geom_boxplot(size=1)+
  theme_bw()+
  ggtitle(t1)
plot(p3)

##### transform for old #####
fcs_info=fcs_info_all%>%dplyr::filter(Subject.Age>60)

# rank transform the vaccine response
fcs_info = fcs_info%>%group_by(STUDY_ACCESSION)%>%
  mutate("0_rank"=rank.ub.average(`0`))%>%
  #group_by(STUDY_ACCESSION,`0_rank`>50)%>%
  mutate("Diff_rank"=rank.ub.average(Diff))%>%
  mutate("28_rank"=rank.ub.average(`28`))%>%
  as.data.frame()



p=ggplot(data=fcs_info,aes(x=STUDY_ACCESSION, y=`28`,color =`28_rank`<=50))+
  geom_jitter(width = 0.2,size=2)+theme_bw()+ylim(2,9)
plot(p)

##### build predictive model for old #####

# read train data 
fcs_info_train = fcs_info%>%
  dplyr::filter(panel_id=="Data_SDY112_CyTOF-1")%>%
  na.omit()

train_data = fcs2DF(fcsFiles=fcs_info_train$new_name,
                    y=(fcs_info_train$`28_rank`>50),
                    assay="CyTOF",
                    b=1/5,
                    fileSampleSize=5000,
                    compFiles=NULL,
                    nameDict=nameDict,
                    excludeTransformParameters=
                      c("FSC-A","FSC-W","FSC-H","Time","Cell_length"))

# parepare data  
common_marker=c("CD4","CCR7","CD3","CD45RA","HLADR")
x_train = pRank(x=train_data[,common_marker],
                xSample=train_data$xSample)
x_train = model.matrix(~.*.,x_train)

# train CytoDx
fit = CytoDx.fit(x=x_train,y=train_data$y,xSample=train_data$xSample,
                 family = "binomial",reg = F)



##### predict in SDY404 FCM for old #####

fcs_info_test = fcs_info%>%
  dplyr::filter(panel_id=="Data_SDY404_FCM-8")%>%
  na.omit()
test_data = fcs2DF(fcsFiles=fcs_info_test$new_name,
                   y=fcs_info_test$`28_rank`>50,
                   assay="FCM",
                   b=1/150,
                   fileSampleSize=5000,
                   compFiles=NULL,
                   nameDict=nameDict,
                   excludeTransformParameters=
                     c("FSC-A","FSC-W","FSC-H","Time","Cell_length"))




# predict using CytoDx 
x_test= pRank(x=test_data[,common_marker],
              xSample=test_data$xSample)

x_test = model.matrix(~.*.,x_test)
pred = CytoDx.pred(fit,xNew=x_test,xSampleNew = test_data$xSample)

# evaluate result
y_test = test_data[,c("xSample","y")]%>%
  group_by(xSample)%>%summarise("y"=unique(y))%>%
  select(-xSample)%>%unlist()
t1 = prediction(predictions = pred$xNew.Pred.sample$y.Pred.s0, 
                labels = y_test)
auc = performance(t1,"auc")@y.values%>%as.numeric()
cat("AUC for older individual = ",auc,"\n")
roc2= performance(t1, measure = "tpr", x.measure = "fpr")

save(test_data,train_data,common_marker,file = "Data/old.rda")

##### plot ROC #####
df1 = data.frame("FP"=roc1@x.values[[1]],"TP"=roc1@y.values[[1]],"group"="young")
df2 = data.frame("FP"=roc2@x.values[[1]],"TP"=roc2@y.values[[1]],"group"="old")
df = rbind(df1,df2)

p=ggplot(df,aes(FP,TP,color=group))+geom_line(size = 2)+
  geom_abline(slope = 1,intercept = 0,size=1,linetype=2)+
  theme_bw()
plot(p)

##### prepare data for 05_test_cellcnn.ipynb#####
fcs_info=fcs_info_all%>%dplyr::filter(Subject.Age>60|Subject.Age<35)

# rank transform the vaccine response
fcs_info = fcs_info%>%group_by(STUDY_ACCESSION)%>%
  mutate("0_rank"=rank.ub.average(`0`))%>%
  #group_by(STUDY_ACCESSION,`0_rank`>50)%>%
  mutate("Diff_rank"=rank.ub.average(Diff))%>%
  mutate("28_rank"=rank.ub.average(`28`))%>%
  as.data.frame()

fcs_info = dplyr::select(fcs_info,Subject.Age,panel_id,STUDY_ACCESSION,
                         new_name,  `0_rank`, `Diff_rank`,`28_rank`)
fcs_info = cbind(fcs_info,
                 "rank_filename"=gsub("Data/HAI_fcs","Result/HAI_rank",fcs_info$new_name))
dir.create("Result/HAI_rank")
write.csv(fcs_info,"Result/HAI_rank/fcs_info.csv",row.names = F)

for (i in 1:nrow(fcs_info)) {
  data_i = fcs2DF(fcsFiles=fcs_info$new_name[i],
                  assay="CyTOF",
                  b=1,
                  fileSampleSize=5000,
                  compFiles=NULL,
                  nameDict=nameDict)
  data_i = data_i[,common_marker]%>%apply(2,rank)/nrow(data_i)*100
  data_i = flowFrame(data_i)
  write.FCS(data_i,filename = as.character(fcs_info$rank_filename[i]))
}

