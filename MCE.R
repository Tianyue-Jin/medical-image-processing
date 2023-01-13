library(mice)
library(VIM)
setwd("C:/Users/Administrator/Desktop/MCE")
data<-read.csv("C:/Users/Administrator/Desktop/MCE/data.csv")
str(data)
summary(data)
options(max.print = 5000)

data[!complete.cases(data),]
sum(is.na(data))
md.pattern(data)
aggr(data,prop=F,numbers=T)
tmp<-mice(data)
data1<-complete(tmp)
stripplot(tmp)
xgbdata<-data1
str(data1)
lassodata<-data1
lassodata<-lassodata[-52]
#####LASSO-Logistic regression
library(glmnet)
x<-model.matrix(Group~.,lassodata)[,-1]
y<-lassodata$Group
lasso.model<-glmnet(x,y,family="binomial",nlambda=100,
                     alpha=1,standardize=T)
plot(lasso.model,xvar = "lambda",label = T)
cv.model<-cv.glmnet(x,y, family="binomial", nlambda=100,
                      alpha=1, standardize=T,maxit=10^5)
plot(cv.model)
cv.model$lambda.min
coef(cv.model, s=cv.model$lambda.min) 
cv.model$lambda.1se
coef(cv.model, s=cv.model$lambda.1se) 
library(rms)
attach(data1)
dd<-datadist(data1)
options(datadist='dd')

library(caret)
set.seed(188928)
devide<-createDataPartition(data1$Group,p=0.7,list=F)
testset<- data1[-devide,]
trainset<-data1[devide,]
testset$tag<-1
trainset$tag<-2
Total<-rbind(testset,trainset)
library(tableone)
x1<-c("Age","Temperature","SBP","DBP","GCS","APACHE","NIHSS","WBC","N","L",
      "NLR","CRP","RDW","TC","TG","HDL","LDL","HCY","Glu","SC","Urea","DD",
      "Fib","CK","CKMB","VOLUME","MLS")
x2<-c("Gender","TOAST","HBP","DM","AF","CHD","Cardiac_insufficiency",
      "His_Stroke","Smoke","Drink","Pneumonia","Ventilation","UTI",
      "Gastrointestinal_bleeding","Hemorrhagictransformation","Consciousness",
      "Seizure","pupil","Gaze","Side","Territory",
      "Lateralventricle","Basalcistern")
table1<-CreateTableOne(vars=c(x1,x2),data=data1,factorVars=x2,
                       strata='Group',addOverall=T)
result1<-print(table1,catDigits=2,contDigits=2,pDigits=3,
               printToggle=T,test=T,showAllLevels=T)
table2<-CreateTableOne(vars=c(x1,x2),data=Total,factorVars=x2,
                       strata='tag',addOverall=T)
result2<-print(table2,catDigits=2,contDigits=2,pDigits=3,
               printToggle=T,test=T,showAllLevels=T)
write.csv(result1,file="C:/Users/Administrator/Desktop/MCE/table1.csv")
write.csv(result2,file="C:/Users/Administrator/Desktop/MCE/table2.csv")

data1[,c("Gender","TOAST","HBP","DM","AF","CHD","Cardiac_insufficiency",
         "His_Stroke","Smoke","Drink","Pneumonia","Ventilation","UTI",
         "Gastrointestinal_bleeding","Hemorrhagictransformation",
         "Consciousness","Seizure","pupil","Gaze","Side","Territory",
         "Lateralventricle","Basalcistern",
         "Group")]<-lapply(data1[,c("Gender","TOAST","HBP","DM","AF","CHD","Cardiac_insufficiency",
                                    "His_Stroke","Smoke","Drink","Pneumonia","Ventilation","UTI",
                                    "Gastrointestinal_bleeding","Hemorrhagictransformation",
                                    "Consciousness","Seizure","pupil","Gaze","Side","Territory",
                                    "Lateralventricle","Basalcistern",
                                    "Group")],factor)

lrm_trainset<-lrm(Group~HBP+AF+Ventilation+GCS+APACHE+NIHSS+
                  WBC+VOLUME+MLS+Basalcistern,
                  data=trainset,x=T,y=T)
lrm_trainset
summary(lrm_trainset)
glm_trainset<-glm(Group~HBP+AF+Ventilation+GCS+APACHE+NIHSS+
                    WBC+VOLUME+MLS+Basalcistern,
                  data=trainset,family=binomial())
summary(glm_trainset)
step(glm_trainset,direction="backward")

lrm2_trainset<-lrm(Group~HBP+AF+Ventilation+APACHE+WBC+VOLUME+MLS,
                   data=trainset,x=T,y=T)
lrm2_trainset
summary(lrm2_trainset)

lrm_testset<-lrm(Group~HBP+AF+Ventilation+APACHE+WBC+VOLUME+MLS,
                 data=testset,x=T,y=T)
lrm_testset
summary(lrm_testset)

trellis.par.set(caretTheme())
cal_trainset<-calibrate(lrm2_trainset,method="boot",B=1000)
cal_trainset
plot(cal_trainset,lwd=2,lty=1,
     xlim=c(0,1),ylim = c(0,1),
     xlab="Predicted Probabiity",
     ylab="Actual Probability",
     col=c(rgb(192,98,82,maxColorValue = 255)))
cal_testset<-calibrate(lrm_testset,method="boot",B=1000)
cal_testset
plot(cal_testset,lwd=2,lty=1,
     xlim=c(0,1),ylim = c(0,1),
     xlab="Predicted Probabiity",
     ylab="Actual Probability",
     col=c(rgb(192,98,82,maxColorValue = 255)))

library(pROC)
library(ResourceSelection)
testset$prediction<-predict(lrm2_trainset,newdata=testset)
roc_int<-roc(testset$DCA,testset$prediction,ci=T)
roc_int
fit_test<-glm(DCA~prediction,data=testset,family=binomial())
hoslem.test(testset$DCA,fitted(fit_test),g=10)
prediction<-predict(fit_test,type='response')
val.prob(prediction,testset$DCA)

library(car)
VIF<-lm(DCA~HBP+AF+Ventilation+GCS+APACHE+NIHSS+
          WBC+VOLUME+MLS+Basalcistern,
        data = trainset)
vif(VIF)
library(rmda)
dca_train<-decision_curve(formula=DCA~HBP+AF+Ventilation+APACHE+WBC+VOLUME+MLS,
                           data=trainset,family="binomial",
                           confidence.intervals=F,bootstraps=10,fitted.risk=F)
dca_test<-decision_curve(formula=DCA~HBP+AF+Ventilation+APACHE+WBC+VOLUME+MLS,
                          data = testset,family="binomial",
                          confidence.intervals=F,bootstraps=10,fitted.risk=F)
list_DCA<-list(dca_train,dca_test)
plot_decision_curve(list_DCA,curve.names =c("Trainning","Validation"),
                    cost.benefit.axis=F,col=c("red","blue"),
                    confidence.intervals=F,standardize=F)

nom<-nomogram(lrm2_trainset,fun=plogis,
              fun.at=c(0.1,0.3,0.5,0.7,0.9),
              lp=F, funlabel = "Incidence of MCE")
plot(nom)

colnames(trainset)
function_list<-c("Group~HBP+AF+Ventilation+APACHE+WBC+VOLUME+MLS",
                 "Group~HBP",
                 "Group~AF",
                 "Group~Ventilation",
                 "Group~APACHE",
                 "Group~WBC",
                 "Group~VOLUME",
                 "Group~MLS")
outcome_name<-"Group"
ROC_names_list<-c("Nomogram Model","HBP","AF",
                  "Ventilation","APACHE","WBC","VOLUME","MLS")
ROC_color_list<-c("red", "#3E606F", "#91AA9D",
                  "#E0D2A3","#EDB17F","#B15646","#007979","#0000C6")
{
  ddist <- datadist(trainset)
  options(datadist='ddist')
  lrm_list_training<-list()
  for(index in 1:length(function_list)){
    lrm_list_training[[index]]<-lrm(as.formula(function_list[index]), data=trainset,x=T,y=T)
  }
  pred_f_training<-list()
  modelroc_training<-list()
  for(index in 1:length(function_list)){
    pred_f_training[[index]]<-predict(lrm_list_training[[index]],trainset)
    modelroc_training[[index]] <- roc(trainset[,outcome_name],pred_f_training[[index]])
  }
  ROC_names_list_training<-ROC_names_list
  for(index in 1:length(function_list)){
    ROC_names_list_training[index]<-paste0(ROC_names_list_training[index]," (AUC=",round(modelroc_training[[index]]$auc[[1]],3),")")
  }
  names(modelroc_training)<-ROC_names_list_training
  roc_list_training<-ggroc(modelroc_training, legacy.axes=TRUE,size = 1)+
    scale_colour_manual(values = ROC_color_list)+
    annotate(geom = "segment",x=0,y=0,xend = 1,yend = 1)+
    theme_bw()
}
plot(roc_list_training)
{
  ddist <- datadist(testset)
  options(datadist='ddist')
  lrm_list_test<-list()
  for(index in 1:length(function_list)){
    lrm_list_test[[index]]<-lrm(as.formula(function_list[index]), data=testset, x=TRUE, y=TRUE,maxit=5000)
  }
  pred_f_test<-list()
  modelroc_test<-list()
  for(index in 1:length(function_list)){
    pred_f_test[[index]]<-predict(lrm_list_test[[index]],testset)
    modelroc_test[[index]] <- roc(testset[,outcome_name],pred_f_test[[index]])
  }
  ROC_names_list_test<-ROC_names_list
  for(index in 1:length(function_list)){
    ROC_names_list_test[index]<-paste0(ROC_names_list_test[index]," (AUC=",round(modelroc_test[[index]]$auc[[1]],3),")")
  }
  names(modelroc_test)<-ROC_names_list_test
  roc_list_test<-ggroc(modelroc_test, legacy.axes=TRUE,size = 1)+
    scale_colour_manual(values = ROC_color_list)+
    annotate(geom = "segment",x=0,y=0,xend = 1,yend = 1)+
    theme_bw()
}
plot(roc_list_test)



######分类树
library(rpart)
library(partykit)
set.seed(987)
treedata<-lassodata
set.seed(188928)
testset.tree<- treedata[-devide,]
trainset.tree<-treedata[devide,]
tree<-rpart(Group~.,data=trainset.tree)
tree$cptable
plotcp(tree)
cp<-min(tree$cptable[3, ])
prune.tree<-prune(tree,cp=cp)
plot(as.party(tree))
plot(as.party(prune.tree))
tree.test<-predict(prune.tree,newdata=testset.tree)
DCA.test<-as.numeric(as.character(testset.tree$Group))
resid<-tree.test-DCA.test
mean(resid^2)
table(tree.test,testset.tree$Group)
library(ggplot2)
ddist.tree1<-datadist(trainset.tree)
options(datadist='ddist.tree1')
ddist.tree2<-datadist(testset.tree)
options(datadist='ddist.tree2')
tree.trainset.pre<-predict(prune.tree,trainset.tree)
roc_tree_train<-roc(trainset.tree$Group,tree.trainset.pre)
tree.roc<-ggroc(roc_tree_train, legacy.axes=TRUE,size = 1)+
  scale_colour_manual(values = ROC_color_list)+
  annotate(geom = "segment",x=0,y=0,xend = 1,yend = 1)+
  theme_bw()
plot(tree.roc)


####极端梯度提升
library(xgboost)
library(Matrix)
xgbdata<-data1
xgb.testset<-xgbdata[-devide,]
xgb.trainset<-xgbdata[devide,]
traindata1<-as.matrix(xgb.trainset[,c(1:50)])
traindata2<-Matrix(traindata1,sparse=T)
outcome_train<-as.numeric(xgb.trainset[,51])
xgb.traindata<-list(data=traindata2,label=outcome_train)
dtrain<-xgb.DMatrix(data=xgb.traindata$data,label=xgb.traindata$label)
testdata1<-as.matrix(xgb.testset[,c(1:50)])
testdata2<-Matrix(testdata1,sparse=T)
outcome_test<-as.numeric(xgb.testset[,51])
xgb.testdata<-list(data=testdata2,label=outcome_test)
dtest<-xgb.DMatrix(data=xgb.testdata$data,label=xgb.testdata$label)
model_xgb<-xgboost(data=dtrain,booster='gbtree',max_depth=6,eta=0.3,
                   objective='binary:logistic',nround=200)
pre_xgb<-predict(model_xgb,newdata=dtest)
prediction_xgb<-ifelse((pre_xgb<=0.5),0,1)
library(caret)
tmp1<-as.factor(prediction_xgb)
tmp2<-as.factor(outcome_test)
xgb.cf<-confusionMatrix(tmp1,tmp2)
xgb.cf
library(DALEX)
prediction_logit<-function(model,x){
  raw_x<-predict(model,x)
  exp(raw_x)/(1+exp(raw_x))
}
logit<-function(x){
  exp(x)/(1+exp(x))
}
explainer_xgb<-explain(model_xgb,data=xgb.traindata$data,
                       y=xgb.traindata$label,
                       predict_function=prediction_logit,link=logit,
                       label="xgboost")
explainer_xgb
varimp<-variable_importance(explainer_xgb,type="raw")
plot(varimp)

imp<-xgb.importance(feature_names=colnames(dtrain),model=model_xgb)
imp
xgb.plot.importance(imp[1:10,])


######SHAP-XGBoost
library(xgboost)
library(SHAPforxgboost)
library(caret)
library(tibble)
library(ggforce)
source("shap.R")
xgbdata<-xgbdata[,c(1:51)]
xgbdata1<-xgbdata[,c(1:50)]
dummy<-dummyVars("~.",data=xgbdata1,fullrank=T)
xgbdata2<-predict(dummy,newdata=xgbdata)
outcome_var<-xgbdata$Group
model.xgb<-xgboost(data=xgbdata2,nround=100,objective="binary:logistic",
                   label=outcome_var)
shap_result<-shap.score.rank(xgb_model=model.xgb,
                             X_train=xgbdata2,
                             shap_approx=F)
shap_result
var_importance(shap_result,top_n=15)
shap_long<-shap.prep(X_train=xgbdata2,top_n=15)
plot.shap.summary(data_long=shap_long)
xgb.plot.shap(data=xgbdata2,model=model.xgb,
              features=names(shap_result$mean_shap_score)[1:15],
              n_col=3,plot_loess=T)
