#Installing of the packages
install.packages("remotes")
remotes:install_github("MrDomani/autofeat")

#Importing the library
library(autofeat)
library(tidyr)


# Reading the data
data_train<-read.csv("C:/Users/Farid/Desktop/SAFE/SAFE_dataset/vehicle_train.csv")
data_test <- read.csv("C:/Users/Farid/Desktop/SAFE/SAFE_dataset/vehicle_test.csv")


X_train <- as.matrix(data_train[,1:ncol(data_train)-1])
y_train <- as.factor(data_train[,ncol(data_train)])

X_test <- as.matrix(data_test[,1:ncol(data_test)-1])
y_test <- as.factor(data_test[,ncol(data_test)])

start.time <- Sys.time()
result <- SAFE(
  X_train,
  y_train,
  X_test,
  y_test,
  operators = list(NULL, list(`+`, `-`, `*`)),
  n_iter = 5,
  nrounds = 5,
  alpha = 0.1,
  gamma = 10,
  bins = 30,
  theta = 0.8
)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
result_write <- as.data.frame.table(result$X_valid)
d1 <- pivot_wider(data = result_write, names_from = "Var2", values_from = "Freq")

write.csv(d1, file = "C:/Users/Farid/Desktop/SAFE/Results/vehicle_valid.csv",row.names=FALSE, na="")



