setwd("C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/Club Mahindra")

traindata <- read.csv("newtraindata.csv", header = T, stringsAsFactors = T)
str(traindata)
testdata <- read.csv("newtestdata.csv", header = T, stringsAsFactors = T)
str(traindata)
traindata=traindata[,-1]
testdata =  testdata[,-1]
str(traindata)




### Random forest

library(randomForest)
rf <- randomForest(amount_spent_per_room_night_scaled~.,data = traindata,
                   do.trace=T,
                   ntree=5,
                   mtry = 100,
                   importance = T)

print(rf)
attributes(rf)
rf$confusion

plot(rf,col = "red")

library(caret)
'''pred<- predict(rf, newtraindata)
head(pred)
head(newtraindata$loan_default)
predi <- ifelse(pred>=0.5,1,0)
tab <- table(predi, traindata$loan_default)
error = 1- sum(diag(tab))/sum(tab)
error*100'''


confusionMatrix(predi,traindata$loan_default)
p1 <- predict(rf,newtestdata)
p1 <- ifelse(p1>=0.5,1,0)
sum(p1)
length(p1)

write.csv(p1,"p8.csv")

str(newtraindata)
t<-tunerf(newtraindata[,-41], newtestdata,
          stepFactor = 0.5,
          plot=T,
          ntreeTry =300,
          trace =T,
          improve= 0.05)


## improved tree
library(randomForest)
rf <- randomForest(loan_default~.,data = newtraindata,
                   ntree=300,
                   mtry = 8,
                   importance = T,
                   proximity =T)
print(rf)

hist(treesize(rf),
     main="No of Nodes for the trees",
     col = "green")


varImpPlot(rf,
           sort = T,
           n.var= 10,
           main= "Top 10 important Variables")
importance(rf)
varUsed(rf)


partialplot(rf,newtraindata,feature,"class(1/2")


## Extract info about single tree

getTree(rf, 1, labelvar =T)


## Multi dimensional scaling plot of proximity matrix
MDSplot(rf, newtraindata$loan_default)
