import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from biokit.viz import corrplot

# load data
dataframe = pd.read_csv("preprocessed_data.csv")
# find correlation
co_relation = dataframe.corr()
# generate correlation plot
cp = corrplot.Corrplot(co_relation)
cp.plot(method='pie', shrink=.9, grid=False)
plt.savefig('correlation.png')
# generate pair plot for all attributes
sns.pairplot(dataframe)
sns.plt.savefig('data_distribution.png')
sns.plt.clf()
# generate heatmap based on correlation
sns.heatmap(co_relation, linewidths=.5, cmap="YlGnBu")
sns.plt.savefig('correlation_heatmap.png')

install.packages('stats');
install.packages('class');
install.packages('rpart');
install.packages('randomForest')
install.packages('ISLR')
install.packages('caret')
install.packages('cvAUC')
install.packages('e1071')
install.packages('ggplot2')
install.packages("gplots")
install.packages("mlbench")
install.packages('adabag')
install.packages('inTrees')


# libraries 
library('stats');
library('class');
library('rpart');
library('randomForest')
library('ISLR')
library('caret')
library('cvAUC')
library('e1071')
library('ggplot2')
library('gplots')
library('gplots')
library('mlbench')
library('adabag')
library('inTrees')
cross_validation_folds = 10;

#Load the data
titantic_data = read.csv('preprocessed_data.csv', header = TRUE, na.strings = c(""));
titantic_data_original = titantic_data

#Summarizing the dataSet
cat("\n=============================== Head of data ==============================================")
cat("\n")
head(titantic_data)
cat("\n===========================================================================================")
cat("\n")
cat("\n")
cat("\n=============================== Summary of data ===========================================")
cat("\n")
summary(titantic_data)
cat("\n===========================================================================================")
cat("\n")
cat("\n")

#Processing data to remove redundant values
titantic_data = unique(titantic_data);

#scaling the data
maxs = apply(titantic_data, MARGIN=2, max);
mins = apply(titantic_data, MARGIN=2, min);
titantic_data = as.data.frame(scale(titantic_data,center = mins,scale = maxs-mins));

# Creating a list for k fold Cross Validation
data_id = sample(1:cross_validation_folds, nrow(titantic_data), replace = TRUE);
list = 1:cross_validation_folds;
Accuracy_SVM = 0;
precision_SVM = 0;
Recall_SVM = 0;
Fmeasure_SVM = 0;
for(i  in 1:cross_validation_folds)
{
  
  trainData  = subset(titantic_data, data_id %in% list[-i]);
  testData = subset(titantic_data, data_id %in% c(i));
  # SVM Building the model
  attach(trainData);
  x = subset(trainData, select = -Survived);
  y = Survived;
  modelSVM = svm(x,y,data= trainData, cost=100, gamma=1, kernel='linear' , type="C-classification");
  detach(trainData);
  #Testing the model
  attach(testData);
  x = subset(testData, select=-Survived);
  y =Survived
  prediction_SVM = predict(modelSVM, x);
  table_SVM=table(prediction_SVM,testData$Survived);
  detach(testData);
  
  #Calculating the metrics for evaluation
  df = data.frame(table_SVM)
  tp <- df[4,3]
  tn <- df[1,3]
  fp <- df[3,3]
  fn <- df[2,3]
  Accuracy_SVM = Accuracy_SVM + ((tp+tn)/(tp+tn+fn+fp));
  precision_SVM = precision_SVM + ((tp/(tp+fp)))
  Recall_SVM= Recall_SVM+(tp/(tp+fn))
}

#Displaying the results
cat("\n=================================== SVM ===================================================")
cat('\nAccuracy of SVM :',100*Accuracy_SVM/cross_validation_folds);
cat('\nPrecision of SVM :',precision_SVM/cross_validation_folds);
cat('\nRecall of SVM :',Recall_SVM/cross_validation_folds);
cat('\nFmeasure of SVM :',(((2*precision_SVM*Recall_SVM)/(precision_SVM+Recall_SVM))/(cross_validation_folds)))
cat("\n===========================================================================================")
cat("\n")
cat("\n")
#scaling the data
maximum = apply(titantic_data, MARGIN=2, max);
minimum = apply(titantic_data, MARGIN=2, min);
titantic_data = as.data.frame(scale(titantic_data,center = minimum,scale = maximum-minimum));

# Creating a list for k fold Cross Validation
sample_data = sample(1:cross_validation_folds, nrow(titantic_data), replace = TRUE);
cross_validation_list = 1:cross_validation_folds;
RandomForest_Accuracy = 0;
KNearestNeighbor_Accuracy=0;
precision_RF = 0;
Recall_RF=0;
precision_KNN = 0;
Recall_KNN=0;

for(i  in 1:cross_validation_folds)
{
  
  titanic_train  = subset(titantic_data, sample_data %in% cross_validation_list[-i]);
  titanic_test = subset(titantic_data, sample_data %in% c(i));
  # Random Forest Model
  titanic_train$Survived = as.factor(titanic_train$Survived);
  titanic_RandomForest_Model = randomForest(Survived~ Age+Embarked+Pclass+Sex+SibSp, titanic_train, importance = TRUE ,replace=TRUE, ntree=300,mtry=1,sampsize=300);
  #varImpPlot(titanic_RandomForest_Model);
  #plot(titanic_RandomForest_Model)
  predict_RandomForest = predict(titanic_RandomForest_Model, titanic_test);
  RandomForest_table = table(predict_RandomForest, titanic_test$Survived);
  dataFrame_RF = data.frame(RandomForest_table)
  
  # Calculating True Positive, False Positive,True Negative, False Negative
  tp_RF <- dataFrame_RF[4,3]
  tn_RF <- dataFrame_RF[1,3]
  fp_RF <- dataFrame_RF[3,3]
  fn_RF <- dataFrame_RF[2,3]
  RandomForest_Accuracy = RandomForest_Accuracy + ((tp_RF+tn_RF)/(tp_RF+tn_RF+fn_RF+fp_RF));
  precision_RF = precision_RF + ((tp_RF/(tp_RF+fp_RF)))
  Recall_RF= Recall_RF+(tp_RF/(tp_RF+fn_RF))
  
  # KNN
  titanic_train$Survived = as.factor(titanic_train$Survived);
  KNN_Model=knn(titanic_train[3:6],titanic_test[3:6],cl=titanic_train$Survived,k=3)
  KNN_table= table(KNN_Model, titanic_test$Survived)
  dataFrame_KNN = data.frame(KNN_table)
  tp_KNN <- dataFrame_KNN[4,3]
  tn_KNN <- dataFrame_KNN[1,3]
  fp_KNN <- dataFrame_KNN[3,3]
  fn_KNN <- dataFrame_KNN[2,3]
  KNearestNeighbor_Accuracy = KNearestNeighbor_Accuracy + ((tp_KNN+tn_KNN)/(tp_KNN+tn_KNN+fn_KNN+fp_KNN));
  precision_KNN = precision_KNN + ((tp_KNN/(tp_KNN+fp_KNN)))
  Recall_KNN= Recall_KNN+(tp_KNN/(tp_KNN+fn_KNN))
}
cat("\n============================== Random Forest ==============================================")
cat('\nAccuracy of Random Forest Model :',100*RandomForest_Accuracy/cross_validation_folds);
cat('\nPrecision of Random Forest Model:',precision_RF/cross_validation_folds);
cat('\nRecall of Random Forest Model:',Recall_RF/cross_validation_folds);
cat('\nFmeasure of Random Forest Model:',(((2*precision_RF*Recall_RF)/(precision_RF+Recall_RF))/(cross_validation_folds)));
cat("\n===========================================================================================")
cat("\n")
cat("\n")
cat("\n================================= KNN =====================================================")
cat('\nAccuracy of KNN Model :',100*KNearestNeighbor_Accuracy/cross_validation_folds);
cat('\nPrecision of KNN Model:',precision_KNN/cross_validation_folds);
cat('\nRecall of KNN Model:',Recall_KNN/cross_validation_folds);
cat('\nFmeasure of KNN Model:',(((2*precision_KNN*Recall_KNN)/(precision_KNN+Recall_KNN))/(cross_validation_folds)))
cat("\n===========================================================================================")

# Creating a list for k fold Cross Validation

data_id = sample(1:cross_validation_folds, nrow(titantic_data), replace = TRUE);
list = 1:cross_validation_folds;

#Setting the accuracy of classifier's to 0 initially
Bagging_accuracy = 0;

#Initializing Precision to 0 for all Classifiers
prec_BG = 0;

# Running the for loop for k cross_validation_folds times for each classifier
for(i  in 1:cross_validation_folds)
{
  trainData  = subset(titantic_data, data_id %in% list[-i]);
  testData = subset(titantic_data, data_id %in% c(i));
  #Bagging
  trainData$Survived = as.factor(trainData$Survived);
  Bagging_model = bagging(Survived~ Embarked+Pclass+Sex, trainData, mfinal= 3, boos = TRUE,rpart.control(maxdepth=10,minsplit=15));
  predict_Bagging = predict.bagging(Bagging_model, testData);
  j = predict_Bagging$confusion;
  df <- data.frame(j)
  tp <- df[4,3]
  tn <- df[1,3]
  fp <- df[3,3]
  fn <- df[2,3]
  Bagging_accuracy = Bagging_accuracy+sum(diag(j))/sum(j);
  prec_BG = prec_BG + mean(100*diag(j)/colSums(j))
  TPR = (tp/(tp+fn))
  FPR = (fp/(fp+tn))
  FMeasure = (2*tp)/((2*tp)+fp+fn)
  Recall_Bg = (tp)/(tp+fn)
}

cat("\n")
cat("\n")
cat("\n====================================== Bagging ============================================")
cat('\nAccuracy of Bagging :',100*Bagging_accuracy/cross_validation_folds);
cat('\nTPR of Bagging :',TPR);
cat('\nFPR of Bagging :',FPR);
cat('\nF-Measure of Bagging :',FMeasure)
cat('\nRecall of Bagging :', Recall_Bg)
cat('\nPrecision of Bagging :',prec_BG/cross_validation_folds)
cat("\n===========================================================================================")

cat("\n")
cat("\n")
cat("\n==================== Best feature extraction rules from Random Forest =====================")
cat("\n")
predictcols <- titantic_data_original[, 2:(ncol(titantic_data_original) )]
titanic_RandomForest_Model = randomForest(Survived~., titantic_data_original, importance = TRUE ,replace=TRUE, ntree=300,mtry=1,sampsize=300);
evaluatingcol <- titantic_data_original[, 1]
finaltreeList <- RF2List(titanic_RandomForest_Model)  # transform rf object to an inTrees' format
rulesextract <- extractRules(finaltreeList, predictcols)  # R-executable conditions
ruleMet <- getRuleMetric(rulesextract,predictcols,evaluatingcol)  # get rule metrics
ruleMet <- selectRuleRRF(ruleMet, predictcols, evaluatingcol)
learner <- buildLearner(ruleMet, predictcols, evaluatingcol)
finalRules <- presentRules(ruleMet, colnames(predictcols))
rules <- finalRules[1:5, ]
print(rules)
cat("\n===========================================================================================")


import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder

# Read the data
data = pd.read_csv('train.csv')

# get number for rows and columns
print("Number of data instances and features/attributes", data.shape)

# Remove nan in 'Embarked' attribute # it has 2 nan values
data.dropna(inplace=True, subset=['Embarked'])

# Build a imputer to replace nan values
# strategy can be : 'mean' or 'median'
imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Apply imputer on 'Age' attribute
data.loc[:, ['Age']] = imputer.fit_transform(data['Age'].values.reshape(-1, 1))

# Build encoder
enc = LabelEncoder()
# modify 'Sex' attribute i.e male and female to 0 and 1
enc.fit(data['Sex'])
data.loc[:, 'Sex'] = enc.transform(data['Sex'])
# modify 'Embarked' attribute i.e  S,C and Q to 0, 1 and 2
enc.fit(data['Embarked'])
data.loc[:,'Embarked'] = enc.transform(data['Embarked'])

# Remove unusable attributes
drop_columns = ['PassengerId','Cabin', 'Name', 'Ticket', 'Fare']
data.drop(drop_columns, axis=1, inplace=True)

# save data in csv
data.to_csv("preprocessed_data.csv", index=False)


