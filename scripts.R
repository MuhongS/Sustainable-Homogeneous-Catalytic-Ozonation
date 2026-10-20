# Re-scale the data from 0 to 1
max <- apply(raw_data, 2, max)
min <- apply(raw_data, 2, min)
data <- as.data.frame(scale(raw_data, center = min, scale = max - min))
data_test <- as.data.frame(scale(raw_data_test, center = min, scale = max - min))

# Define 5 fold repeated cross validation 
cv_control <- trainControl(method = "repeatedcv", number = 5,  repeats = 15, search = "grid")

#'@LinearRegression #
set.seed(999)
# Define the custom grid
linear_grid <- expand.grid(alpha = seq(0, 1, by = 0.1), lambda = seq(0.001, 0.1, length = 10))
# Running the model
linear_model <- train(as.formula(Efficiency ~ .), data = data, method = "glmnet", trControl = cv_control, tuneGrid = linear_grid, metric = "RMSE")
# Check the best parameter 
print(linear_model)
# Check accuracy of the model by comparing with 20 existing test set
Prediction_Linear <- predict(linear_model, newdata = data_test)
# Check accuracy of the model by comparing with 60 existing train set
Prediction_Linear_trained <- predict(linear_model, newdata = data)
# Check for RMSE, R2 and MAE
final_metrics_linear <- postResample(pred = Prediction_Linear, obs = data_test$Efficiency)
final_metrics_linear_trained <- postResample(pred = Prediction_Linear_trained, obs = data$Efficiency)

#'@DecisionTree #
set.seed(999)
# Define the custom grid
tree_grid <- expand.grid(cp = seq(0, 0.1, by = 0.005))
# Running the model
tree_model <- train(as.formula(Efficiency ~ .), data = data, method = "rpart", trControl = cv_control, tuneGrid = tree_grid, metric = "RMSE")
# Check the best parameter 
print(tree_model)
# Check accuracy of the model by comparing with 20 existing test set
Prediction_tree <- predict(tree_model, newdata = data_test)
# Check accuracy of the model by comparing with 60 existing train set
Prediction_tree_trained <- predict(tree_model, newdata = data)
# Check for RMSE, R2 and MAE
final_metrics_tree <- postResample(pred = Prediction_tree, obs = data_test$Efficiency)
final_metrics_tree_trained <- postResample(pred = Prediction_tree_trained, obs = data$Efficiency)

#'@RandomForest #
set.seed(999)
# Define the custom grid
rf_grid <- expand.grid(mtry = c(2, 3, 4)) 
# Running the model
rf_model <- train(as.formula(Efficiency ~ .), data = data,method = "rf", trControl = cv_control, tuneGrid = rf_grid,metric = "RMSE", ntree = 500)
# Check the best parameter 
print(rf_model)
# Check accuracy of the model by comparing with 20 existing test set
Prediction_rf <- predict(rf_model, newdata = data_test)
# Check accuracy of the model by comparing with 60 existing train set
Prediction_rf_trained <- predict(rf_model, newdata = data)
# Check for RMSE, R2 and MAE
final_metrics_rf <- postResample(pred = Prediction_rf, obs = data_test$Efficiency)
final_metrics_rf_trained <- postResample(pred = Prediction_rf_trained, obs = data$Efficiency)

#'@ExtremeGradientBoosting #
set.seed(999)
# Define the custom grid
xgb_grid <- expand.grid(nrounds = c(50, 100, 150), max_depth = c(2, 3, 4), eta = c(0.01, 0.05, 0.1), gamma = c(0, 1), colsample_bytree = c(0.8, 1.0), min_child_weight = 1, subsample = c(0.8, 1.0))
# Running the model
xgb_model <- train(as.formula(Efficiency ~ .), data = data, method = "xgbTree", trControl = cv_control, tuneGrid = xgb_grid, metric = "RMSE", verbosity = 0)
# Check the best parameter
print(xgb_model)
# Check accuracy of the model by comparing with 20 existing test set
Prediction_xgb <- predict(xgb_model, newdata = data_test)
# Check accuracy of the model by comparing with 60 existing train set
Prediction_xgb_trained <- predict(xgb_model, newdata = data)
# Check for RMSE, R2 and MAE
final_metrics_xgb <- postResample(pred = Prediction_xgb, obs = data_test$Efficiency)
final_metrics_xgb_trained <- postResample(pred = Prediction_xgb_trained, obs = data$Efficiency)

#'@K-NearestNeighbour #
set.seed(999)
# Define the custom grid
knn_grid <- expand.grid(k = seq(2, 15, by = 1)) 
# Running the model
knn_model <- train(as.formula(Efficiency ~ .), data = data, method = "knn", trControl = cv_control, tuneGrid = knn_grid, metric = "RMSE")    
#Check the best parameter
print(knn_model)
# Check accuracy of the model by comparing with 20 existing test set
Prediction_knn <- predict(knn_model, newdata = data_test)
# Check accuracy of the model by comparing with 60 existing train set
Prediction_knn_trained <- predict(knn_model, newdata = data)
# Check for RMSE, R2 and MAE
final_metrics_knn <- postResample(pred = Prediction_knn, obs = data_test$Efficiency)
final_metrics_knn_trained <- postResample(pred = Prediction_knn_trained, obs = data$Efficiency)

#'@ArtificialNeuralNetwork #
set.seed(999)
# Define the custom grid
ann_grid <- expand.grid(size = c(1, 2, 3, 4, 5), decay = c(0.001, 0.01, 0.05, 0.1))      
# Running the model
ann_model <- train(as.formula(Efficiency ~ .), data = data, method = "nnet", trControl = cv_control, tuneGrid = ann_grid, metric = "RMSE", trace = FALSE, maxit = 500, linout = TRUE) 
#Check the best parameter
print(ann_model)
# Check accuracy of the model by comparing with 20 existing test set
Prediction_ann <- predict(ann_model, newdata = data_test)
# Check accuracy of the model by comparing with 60 existing train set
Prediction_ann_trained <- predict(ann_model, newdata = data)
# Check for RMSE, R2 and MAE
final_metrics_ann <- postResample(pred = Prediction_ann, obs = data_test$Efficiency)
final_metrics_ann_trained <- postResample(pred = Prediction_ann_trained, obs = obs = data$Efficiency)
