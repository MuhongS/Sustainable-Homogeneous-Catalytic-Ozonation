# Re-scale the data from 0 to 1 
max <- apply(data, 2, max)
min <- apply(data, 2, min)
data <- as.data.frame(scale(data, center = min, scale = max - min))

# Define 5 fold repeated cross validation 
cv_control <- trainControl(method = "repeatedcv", number = 5,  repeats = 15, search = "grid")

#'@LinearRegression #
set.seed(999)
# Define the custom grid
linear_grid <- expand.grid(alpha = seq(0, 1, by = 0.1), lambda = seq(0.001, 0.1, length = 10))
# Running the model
linear_model <- train(as.formula(Efficiency ~ .), data = data, method = "glmnet", trControl = cv_control, tuneGrid = linear_grid, metric = "RMSE")

#'@DecisionTree #
set.seed(999)
# Define the custom grid
tree_grid <- expand.grid(cp = seq(0, 0.1, by = 0.005))
# Running the model
tree_model <- train(as.formula(Efficiency ~ .), data = data, method = "rpart", trControl = cv_control, tuneGrid = tree_grid, metric = "RMSE")

#'@RandomForest #
set.seed(999)
# Define the custom grid
rf_grid <- expand.grid(mtry = c(2, 3, 4)) 
# Running the model
rf_model <- train(as.formula(Efficiency ~ .), data = data,method = "rf", trControl = cv_control, tuneGrid = rf_grid,metric = "RMSE", ntree = 500)

#'@ExtremeGradientBoosting #
set.seed(999)
# Define the custom grid
xgb_grid <- expand.grid(nrounds = c(50, 100, 150), max_depth = c(2, 3, 4), eta = c(0.01, 0.05, 0.1), gamma = c(0, 1), colsample_bytree = c(0.8, 1.0), min_child_weight = 1, subsample = c(0.8, 1.0))
# Running the model
xgb_model <- train(as.formula(Efficiency ~ .), data = data, method = "xgbTree", trControl = cv_control, tuneGrid = xgb_grid, metric = "RMSE", verbosity = 0)

#'@K-NearestNeighbour #
set.seed(999)
# Define the custom grid
knn_grid <- expand.grid(k = seq(2, 15, by = 1)) 
# Running the model
knn_model <- train(as.formula(Efficiency ~ .), data = data, method = "knn", trControl = cv_control, tuneGrid = knn_grid, metric = "RMSE")    

#'@ArtificialNeuralNetwork #
set.seed(999)
# Define the custom grid
ann_grid <- expand.grid(size = c(1, 2, 3, 4, 5), decay = c(0.001, 0.01, 0.05, 0.1))      
# Running the model
ann_model <- train(as.formula(Efficiency ~ .), data = data, method = "nnet", trControl = cv_control, tuneGrid = ann_grid, metric = "RMSE", trace = FALSE, maxit = 500, linout = TRUE) 
