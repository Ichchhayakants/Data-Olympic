setwd("C:/Users/TheUnlikelyMonk/Desktop/Analytics Vidhya/Club Mahindra")

traindata <- read.csv("train.csv", header = T, stringsAsFactors = T)
testdata <- read.csv("test.csv", header = T, stringsAsFactors = T)

newtraindata<- traindata
newtestdata<- testdata

newtraindata$reservation_id<- as.numeric(newtraindata$reservation_id)
newtraindata$booking_date <- as.Date(newtraindata$booking_date, format = "%d/%m/%y")
newtraindata$checkin_date <- as.Date(newtraindata$checkin_date, format = "%d/%m/%y")
newtraindata$checkout_date<- as.Date(newtraindata$checkout_date, format = "%d/%m/%y")
newtraindata$booking_checkin <- as.numeric(newtraindata$checkin_date-newtraindata$booking_date)
str(newtraindata)
newtraindata <- newtraindata[,-c(2:4)]
newtraindata$member_age_buckets<- as.numeric(newtraindata$member_age_buckets)
newtraindata$memberid <-  as.numeric(newtraindata$memberid)
newtraindata$cluster_code <- as.numeric(newtraindata$cluster_code)
newtraindata$reservationstatusid_code <- as.numeric(newtraindata$reservationstatusid_code)
newtraindata$resort_id <-  as.numeric(newtraindata$resort_id)
str(newtraindata)


newtestdata$reservation_id<- as.numeric(newtestdata$reservation_id)
newtestdata$booking_date <- as.Date(newtestdata$booking_date, format = "%d/%m/%y")
newtestdata$checkin_date <- as.Date(newtestdata$checkin_date, format = "%d/%m/%y")
newtestdata$checkout_date<- as.Date(newtestdata$checkout_date, format = "%d/%m/%y")
newtestdata$booking_checkin <-as.numeric(newtestdata$checkin_date-newtestdata$booking_date)
newtestdata <- newtestdata[,-c(2:4)]
newtestdata$member_age_buckets<- as.numeric(newtestdata$member_age_buckets)
newtestdata$memberid <-  as.numeric(newtestdata$memberid)
newtestdata$cluster_code <- as.numeric(newtestdata$cluster_code)
newtestdata$reservationstatusid_code <- as.numeric(newtestdata$reservationstatusid_code)
newtestdata$resort_id <-  as.numeric(newtestdata$resort_id)
str(newtestdata)

nrow(newtraindata)
nrow(newtestdata)
sum(is.na(newtraindata))
sum(is.na(newtestdata))


# Imputation
replaceNAsWithMean <- function(x){
  replace(x,is.na(x),mean(x[!is.na(x)]))
}

newtraindata <- replaceNAsWithMean(newtraindata)
newtestdata <- replaceNAsWithMean(newtestdata)

normalize<-function(x){
  (x- min(x)) /(max(x)-min(x))
}
traindata_scaled<-as.data.frame(lapply(newtraindata, normalize))
testdata_scaled<-as.data.frame(lapply(newtestdata, normalize))

write.csv(newtraindata,file = "traindata_scaled.csv")
write.csv(newtestdata, file = "testdata_scaled.csv")

write.csv(newtraindata,file = "newtraindata.csv")
write.csv(newtestdata, file = "newtestdata.csv")



lmMod <- lm(amount_spent_per_room_night_scaled~., data=newtraindata)  # build the model

distPred <- predict(lmMod,newtestdata)  # predict distance

distPred<- as.data.frame(distPred)

distPred <- as.data.frame(cbind(testdata$reservation_id,distPred))
colnames(distPred)<- c("reservation_id","amount_spent_per_room_night_scaled")

write.csv(distPred, file = "Out1.csv")

