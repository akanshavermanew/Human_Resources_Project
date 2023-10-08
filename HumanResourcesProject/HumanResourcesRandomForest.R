#Human Resource project done using random forest 
#AUC Score is 0.84
#The more auc score the better model is

library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(tidyr)
library(ROCit)
library(vip)
library(rpart.plot)
library(DALEXtra)

setwd("F:/R study materials/Projects/HumanResourcesProject")
## ----
hrp_train=read.csv("hr_train.csv",stringsAsFactors = FALSE)
hrp_test=read.csv("hr_test.csv",stringsAsFactors = FALSE)

glimpse(hrp_train)

vis_dat(hrp_train)

unique(hrp_train$salary)

#Since we will be using random forest we need to convert data type of response (which is store in this case) to factor type using function as.factor. This is how randomforest differentiates from regression & classification.If we need to build a regression model then response variable should be kept numeric else factor for classification.


#for classification if we are using random forest algo then we need to convert the output variable into factor 
#and for linear regression we need to make it as numeric
#only as.factor was giving boolean values so added as.integer as well as as.logical

hrp_train$left=as.factor((as.integer(as.logical((hrp_train$left)))))

dp_pipe=recipe(left ~ .,data=hrp_train) %>% 
  update_role(sales,salary,new_role="to_dummies") %>% 
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>% 
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data = NULL)
test=bake(dp_pipe,new_data=hrp_test)

vis_dat(train)
glimpse(train)

summary(dp_pipe)

#building random forest
rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

folds = vfold_cv(train, v = 5)

rf_grid = grid_regular(mtry(c(5,19)), trees(c(100,500)),     #for random forest passing some values is compulsory otherwise it will give error in decision tree the case was not like this
                       min_n(c(2,10)),levels = 3)

#IMP BELOW 2 STEPS
# c(5,19)  means start with 5 and go till 19
# mtry values should be <= features in your table
my_res=tune_grid(
  rf_model,
  left~.,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)

autoplot(my_res)+theme_light()


fold_metrics=collect_metrics(my_res)    #grid search it will show table with parameters like mtry,tree_depth,etc (parameters of whatever model you are using

my_res %>% show_best()         #it will give best roc_auc value for our model from grid serch with all parameters of whatever model you are using like mtry,tree_depth,etc

#Finalizing the model
final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res,"roc_auc")) %>% 
  fit(left~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predicitons

train_pred=predict(final_rf_fit,new_data = train,type="prob") %>% select(.pred_1)
test_pred=predict(final_rf_fit,new_data = test,type="prob") %>% select(.pred_1)

#writing in csv file
write.table(test_pred,"Akansha_Verma_P4_part2.csv",row.names=F,col.names = "left")
getwd()




#Quiz
---------------------------------------------------------------------------------------
#q1)Find out total promotions happened in last 5 years
  
  sum(hrp_train$promotion_last_5years)

#Ans-228
-----------------------------------------------------------------------------------
---------------------------------------------------------------------------------------
#q2)Find out the variance in statisfaction_level for category 0 of variable ‘left’ (round off to 4 decimal places).
  
# Assuming you have a dataset or data frame called 'data' that contains the relevant variables
  
# Subset the data to include only category 0 of 'left'
category_0_data <- subset(hrp_train, left == 0)

# Calculate the variance of 'satisfaction_level'
variance <- var(category_0_data$satisfaction_level)

# Round off the variance to 4 decimal places
rounded_variance <- round(variance, 4)

#Ans - 0.0487
------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------
#q3)According to given data what is the probability that someone will leave the organisation if they were involved in a work accident? (round off 2 decimal places)
  
  # Assuming you have a dataset or data frame called 'data' that contains the relevant variables
  
  # Calculate the total number of employees involved in work accidents
  total_accident_employees <- sum(hrp_train$work_accident == 1)

# Calculate the total number of employees who left the organization after a work accident
left_after_accident <- sum(hrp_train$work_accident == 1 & hrp_train$left == 1)

# Calculate the probability of leaving after a work accident
probability_leave_after_accident <- round(left_after_accident / total_accident_employees, 2)

#Ans - NAN i.e. =0
------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------
#q4)Find out which category of salary has maximum employee resignation.
#Note: you need to write name of just that category .
  
  # Assuming you have a dataset or data frame called 'data' that contains the relevant variables
  
  # Create a table summarizing the count of resignations for each salary category
  resignations_by_salary <- table(hrp_train$left, hrp_train$salary)

# Find the category of salary with the maximum resignations
max_resignation_category <- names(which.max(resignations_by_salary))


#Ans- low
------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------
#q5)What is the median time spent with the company among people leaving the company?
  
  # Assuming you have a dataset or data frame called 'data' that contains the relevant variables
  
  # Subset the data to include only employees who are leaving the company
  leaving_data <- subset(hrp_train, left == 1)

# Calculate the median time spent with the company among people leaving
median_time_leaving <- median(leaving_data$time_spend_company)

#Ans - 3