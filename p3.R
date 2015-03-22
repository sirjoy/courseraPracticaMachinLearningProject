rm(list=ls())

library("caret")

path.base <- '/home/sirjoy/Dropbox/CursosRecibidos/Coursera/PracticalMachineLearning_2014/Project/'
path.training <- paste0(path.base, 'pml-training.csv')
path.testing <- paste0(path.base, 'pml-testing.csv')

trainingRaw <- read.csv(path.training)
testingRaw <- read.csv(path.testing)

# Number of NA per feature: cero or many
numNA <- sapply(names(trainingRaw), function(x) sum(is.na(trainingRaw[,x])))
plot(numNA)
completeFeatures <- names(trainingRaw)[numNA == 0] #Features without NA
numericFeatures <- names(trainingRaw)[sapply(trainingRaw, is.numeric)]

# 7 first features no usefull for classification
completeFeatures <- completeFeatures[8:length(completeFeatures)]
selectedFeatures <- completeFeatures[completeFeatures %in% numericFeatures]
data <- trainingRaw[, selectedFeatures] 
clases <- trainingRaw$classe

# Partition of training raw data in sets for training and validation
set.seed(838)
indexTraining <- createDataPartition(clases, p = 0.2, list = F, times = 1)
training <- data[indexTraining,]
trainingLabels <- clases[indexTraining]
testing <- data[-indexTraining,]
testingLabels <- clases[-indexTraining]

prePCA <- preProcess(training, method='pca', thresh=0.8)
prePCA
trainingPCA <- predict(prePCA, training)

## 10-fold CV
fitControl <- trainControl(method = "cv",  number = 10)
model <- train(x = trainingPCA, y = trainingLabels, method='rf', trControl = fitControl)
model
#featurePlot(training, y = clases, plot = 'pairs', auto.key = list(columns = 3))
testingPCA <- predict(prePCA, testing)
testPC <- predict(model, testingPCA)
confusionMatrix(testPC, testingLabels)

path.base <- '/home/sirjoy/Dropbox/CursosRecibidos/Coursera/PracticalMachineLearning_2014/Project/'
save(model,file= paste0(path.base, 'modelAllTrainingDataPCA_cv.Rdata'))

testingFinal <- testingRaw[,selectedFeatures]
testingFinalPCA <- predict(prePCA, testingFinal)
testFinalPred <- predict(model, testingFinalPCA)
#confusionMatrix(testingFinalPred, )

setwd(paste0(path.base, 'answers'))
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

head(testFinalPred)
pml_write_files(testFinalPred)
