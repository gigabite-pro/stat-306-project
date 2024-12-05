#install.packages(readr)
#library(readr)

rank <- read.csv("/Users/chin2/Desktop/Stat306/Final Project/data.csv",
                 header = TRUE)

rank <- as.data.frame(rank)
head(rank)

# Load the packages
library(dplyr)
library(ggplot2)

# Step 1: Wrangle the data
rank_wrangled <- rank %>%
  dplyr::select(
    Location,
    No.of.student,
    No.of.student.per.staff,
    International.Student,
    Female.Male.Ratio,
    OverAll.Score
  ) %>%
  
  # Filter only relevant locations
  filter(Location %in% c("Canada", "United States", "United Kingdom")) %>%
  
  mutate(
    # Handle ranges in 'OverAll.Score'
    OverAll.Score = ifelse(
      grepl("–|-", OverAll.Score),
      sapply(strsplit(as.character(OverAll.Score), "[–-]"), function(x)
        mean(as.numeric(x))),
      as.numeric(OverAll.Score)
    ),
    
    Location = as.factor(Location),
    
    # Remove commas from 'No.of.student' and convert to numeric
    No.of.student = as.numeric(gsub(",", "", No.of.student)),
    
    # Remove % sign from 'International.Student' and convert to numeric
    International.Student = as.numeric(gsub("%", "", International.Student)),
    
    # Convert 'No.of.student.per.staff' to numeric
    No.of.student.per.staff = as.numeric(No.of.student.per.staff),
    
    # Convert 'Female.Male.Ratio' from "x : y" to numeric ratio x/y
    Female.Male.Ratio = as.numeric(sub(":.*", "", Female.Male.Ratio)) /
      as.numeric(sub(".*:", "", Female.Male.Ratio))
  ) %>%
  
  # Drop rows with missing or invalid values
  drop_na()

# View the processed data
summary(rank_wrangled)

#  Distribution of Overall Score:
ggplot(rank_wrangled, aes(x = OverAll.Score)) +
  geom_histogram(binwidth = 2, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Overall Score", x = "Overall Score", y = "Frequency") +
  theme_minimal()

# Boxplots for Each Variable by Location
# Boxplot for Overall Score by Location
ggplot(rank_wrangled, aes(x = Location, y = OverAll.Score, fill = Location)) +
  geom_boxplot() +
  labs(title = "Overall Score by Location", y = "Overall Score", x = "Location") +
  theme_minimal()

# Boxplot for Number of Students by Location
ggplot(rank_wrangled, aes(x = Location, y = No.of.student, fill = Location)) +
  geom_boxplot() +
  labs(title = "Number of Students by Location", y = "Number of Students", x = "Location") +
  theme_minimal()

# Boxplot for Student-to-Staff Ratio by Location
ggplot(rank_wrangled, aes(x = Location, y = No.of.student.per.staff, fill = Location)) +
  geom_boxplot() +
  labs(title = "Student-to-Staff Ratio by Location", y = "Student-to-Staff Ratio", x = "Location") +
  theme_minimal()

# Boxplot for International Student Percentage by Location
ggplot(rank_wrangled, aes(x = Location, y = International.Student, fill = Location)) +
  geom_boxplot() +
  labs(title = "International Students by Location", y = "Percentage of International Students", x = "Location") +
  theme_minimal()

# Plots
# Create a bar plot of Location frequencies
ggplot(rank_wrangled, aes(x = Location)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(title = "Frequency of Locations", x = "Location", y = "Frequency") +
  theme_minimal()




# Perform forward selection
library(leaps)
# Perform forward selection
forward_model <- regsubsets(
  OverAll.Score ~ No.of.student + No.of.student.per.staff + International.Student +
    Female.Male.Ratio + Location,
  data = rank_wrangled,
  method = "forward"  # Specify forward selection method
)

# Summarize the model results
model_summary <- summary(forward_model)

# View results
model_summary

# Find the best model based on Adjusted R-squared
best_model_index <- which.max(model_summary$adjr2)

# View the best model's predictors
best_model_predictors <- model_summary$which[best_model_index, ]
best_model_predictors

# Fit the final model
final_model <- lm(
  OverAll.Score ~ No.of.student + No.of.student.per.staff + International.Student + Female.Male.Ratio,
  data = rank_wrangled
)

# Summary of the final model
summary(final_model)

# Calculate residuals
residuals <- residuals(final_model)

# Plot the residuals
ggplot(data.frame(Predicted = fitted(final_model), Residuals = residuals), aes(x = Predicted, y = Residuals)) +
  geom_point(color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residual Plot", x = "Predicted Values", y = "Residuals") +
  theme_minimal()



# Fit the model with log-transformed dependent variable
final_model_log <- lm(
  log(OverAll.Score) ~ No.of.student + No.of.student.per.staff + International.Student + Female.Male.Ratio,
  data = rank_wrangled
)

# Summary of the transformed model
summary(final_model_log)

# Calculate residuals for the new model
residuals_log <- residuals(final_model_log)

# Plot the new residuals
ggplot(data.frame(Predicted = fitted(final_model_log), Residuals = residuals_log), aes(x = Predicted, y = Residuals)) +
  geom_point(color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residual Plot (Log Transformation)", x = "Predicted Values", y = "Residuals") +
  theme_minimal()


library(caret)

set.seed(123)  # Set a random seed for reproducibility
trainIndex <- createDataPartition(rank_wrangled$OverAll.Score, p = 0.8, list = FALSE)
train_data <- rank_wrangled[trainIndex, ]
test_data <- rank_wrangled[-trainIndex, ]

# Step 3: Make predictions on the test set
test_predictions_log <- predict(final_model_log, newdata = test_data)

# Step 4: Evaluate the model
# We need to reverse the log transformation for the predictions to compare with actual values
test_predictions <- exp(test_predictions_log)

# Calculate the RMSE (Root Mean Squared Error) to evaluate performance
rmse_value <- sqrt(mean((test_data$OverAll.Score - test_predictions)^2))
rmse_value

# Plot actual vs predicted values
ggplot(data.frame(Actual = test_data$OverAll.Score, Predicted = test_predictions), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red") +  # Line of perfect prediction
  labs(title = "Actual vs Predicted (Log-Transformed Model)", x = "Actual Values", y = "Predicted Values") +
  theme_minimal()

# Set up cross-validation with 10 folds (you can adjust the number of folds as needed)
train_control <- trainControl(method = "cv", number = 10)

# Train the model using cross-validation
cv_model <- train(
  log(OverAll.Score) ~ No.of.student + No.of.student.per.staff + International.Student + Female.Male.Ratio,
  data = rank_wrangled,
  method = "lm",          # Use linear regression
  trControl = train_control
)

# View cross-validation results
cv_model

exp_rmse <- exp(0.2368)
exp_mae <- exp(0.1878)

exp_rmse
exp_mae


