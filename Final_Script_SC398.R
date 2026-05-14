library(randomForest)
library(modeldata)
library(lime)
library(fastshap)
library(shapviz)
library(iml)
library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(caret)

set.seed(123)

expedition_data <- read.csv("Desktop/Final_Data/expeditions.csv")
peak_data       <- read.csv("Desktop/Final_Data/peaks.csv")
member_data     <- read.csv("Desktop/Final_Data/members.csv")

# ── Data Preparation ──────────────────────────────────────────────────────────

merged_data <- member_data %>%
  left_join(peak_data %>% select(peak_id, height_metres), by = "peak_id") %>%
  left_join(expedition_data %>% select(expedition_id, year), by = "expedition_id")

df_model <- merged_data %>%
  select(success, age, sex, season, year, oxygen_used, hired, solo, height_metres)

df_model_clean <- df_model %>%
  filter(!is.na(age), !is.na(sex), sex != "Unknown", sex != "") %>%
  mutate(
    success      = as.factor(success),
    sex          = as.factor(sex),
    season       = as.factor(season),
    oxygen_used  = as.factor(oxygen_used),
    hired        = as.factor(hired),
    solo         = as.factor(solo)
  ) %>%
  na.omit()

print(table(df_model_clean$success))

# ── Train / Test Split ────────────────────────────────────────────────────────

train_index <- sample(seq_len(nrow(df_model_clean)), size = 0.8 * nrow(df_model_clean))
train_data  <- df_model_clean[train_index, ]
test_data   <- df_model_clean[-train_index, ]

# ── 5-Fold Stratified Cross-Validation ───────────────────────────────────────

class_counts  <- table(train_data$success)
class_weights <- max(class_counts) / class_counts

cv_folds <- createFolds(train_data$success, k = 5, returnTrain = TRUE)

cv_accuracies <- sapply(cv_folds, function(idx) {
  fold_train <- train_data[idx, ]
  fold_val   <- train_data[-idx, ]
  fold_model <- randomForest(
    success ~ .,
    data       = fold_train,
    ntree      = 500,
    classwt    = class_weights,
    importance = FALSE
  )
  preds <- predict(fold_model, fold_val)
  mean(preds == fold_val$success)
})

cat("5-Fold CV Accuracy:", round(mean(cv_accuracies), 4),
    "± SD:", round(sd(cv_accuracies), 4), "\n")

# ── Final Model (Balanced Class Weights) ─────────────────────────────────────

rf_model <- randomForest(
  success ~ .,
  data       = train_data,
  ntree      = 500,
  classwt    = class_weights,
  importance = TRUE
)

varImpPlot(rf_model, main = "Standard Random Forest Importance")

# ── Exploratory Visualisations ────────────────────────────────────────────────

plot_success <- ggplot(df_model_clean, aes(x = success, fill = success)) +
  geom_bar(alpha = 0.8, color = "black", width = 0.6) +
  scale_fill_manual(values = c("FALSE" = "#E69F00", "TRUE" = "#009E73")) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  labs(title    = "Expedition Success Distribution",
       subtitle = "Distribution of target variable (Individual Member Success)",
       x = "Summit Success", y = "Number of Climbers") +
  theme_minimal() + theme(legend.position = "none")

plot_season <- df_model_clean %>%
  group_by(season, success) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(season) %>%
  mutate(prop = count / sum(count)) %>%
  filter(success == "TRUE") %>%
  ggplot(aes(x = reorder(season, -prop), y = prop, fill = season)) +
  geom_col(alpha = 0.8, color = "black") +
  scale_y_continuous(labels = percent_format()) +
  labs(title    = "Success Rate by Season",
       subtitle = "Percentage of climbers who successfully summited",
       x = "Season", y = "Success Rate") +
  theme_minimal() + theme(legend.position = "none")

plot_age <- ggplot(df_model_clean, aes(x = age, fill = success)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("FALSE" = "#E69F00", "TRUE" = "#009E73")) +
  labs(title    = "Age Distribution by Climbing Outcome",
       subtitle = "Density of climber ages separated by summit success",
       x = "Age", y = "Density", fill = "Success") +
  theme_minimal()

plot_oxygen_height <- ggplot(df_model_clean, aes(x = oxygen_used, y = height_metres, fill = success)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.2) +
  scale_fill_manual(values = c("FALSE" = "#E69F00", "TRUE" = "#009E73")) +
  labs(title    = "Peak Height vs. Oxygen Usage",
       subtitle = "Summit success mapped to target elevation and supplemental oxygen",
       x = "Supplemental Oxygen Used", y = "Peak Height (Metres)", fill = "Success") +
  theme_minimal()

plot_hired <- ggplot(df_model_clean, aes(x = hired, fill = success)) +
  geom_bar(position = "fill", alpha = 0.8, color = "black", width = 0.6) +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_manual(values = c("FALSE" = "#E69F00", "TRUE" = "#009E73")) +
  labs(title    = "Success Rate by Hired Status",
       subtitle = "Relative summit success rates for hired vs non-hired climbers",
       x = "Is Staff Hired?", y = "Proportion", fill = "Success") +
  theme_minimal()

plot_year <- df_model_clean %>%
  group_by(year, success) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(year) %>%
  mutate(prop = count / sum(count)) %>%
  filter(success == "TRUE") %>%
  ggplot(aes(x = year, y = prop)) +
  geom_line(color = "#009E73", linewidth = 0.8) +
  geom_smooth(method = "loess", se = TRUE, color = "#4C72B0", alpha = 0.2) +
  scale_y_continuous(labels = percent_format()) +
  labs(title    = "Summit Success Rate Over Time",
       subtitle = "Annual proportion of successful summits (1905–2019)",
       x = "Year", y = "Success Rate") +
  theme_minimal()

print(plot_success)
print(plot_season)
print(plot_age)
print(plot_oxygen_height)
print(plot_hired)
print(plot_year)

# ── Global SHAP Values ────────────────────────────────────────────────────────

pfun <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "TRUE"]
}

X_explain <- test_data %>% select(-success)

set.seed(42)
shap_values <- fastshap::explain(
  rf_model,
  X            = X_explain,
  pred_wrapper = pfun,
  nsim         = 50,
  adjust       = TRUE
)

shp <- shapviz(shap_values, X = X_explain)

plot_shap_beeswarm <- sv_importance(shp, kind = "beeswarm") +
  ggtitle("SHAP Summary: Feature Impact on Summit Success") +
  theme_minimal()

plot_shap_bar <- sv_importance(shp, kind = "bar", fill = "#4C72B0") +
  ggtitle("Global Feature Importance (Mean |SHAP|)") +
  theme_minimal()

plot_shap_height <- sv_dependence(shp, v = "height_metres", color_var = "oxygen_used") +
  ggtitle("SHAP Dependence: Peak Height × Oxygen Use") +
  theme_minimal()

plot_shap_age <- sv_dependence(shp, v = "age", color_var = "hired") +
  ggtitle("SHAP Dependence: Age × Hired Status") +
  theme_minimal()

print(plot_shap_bar)
print(plot_shap_beeswarm)
print(plot_shap_height)
print(plot_shap_age)

# ── Uncertainty Quantification (Conformal Prediction) ─────────────────────────
# Split-conformal approach: calibrate on a held-out calibration set

set.seed(99)
cal_idx   <- sample(seq_len(nrow(test_data)), size = floor(0.5 * nrow(test_data)))
cal_data  <- test_data[cal_idx, ]
eval_data <- test_data[-cal_idx, ]

cal_probs  <- predict(rf_model, cal_data, type = "prob")[, "TRUE"]
cal_labels <- as.numeric(as.character(cal_data$success)) == 1

# Non-conformity scores: 1 - predicted prob for the true class
cal_scores <- ifelse(cal_labels, 1 - cal_probs, cal_probs)

alpha       <- 0.10   # target miscoverage
q_threshold <- quantile(cal_scores, probs = 1 - alpha)

eval_probs  <- predict(rf_model, eval_data, type = "prob")[, "TRUE"]
pred_set_includes_true <- (1 - eval_probs) <= q_threshold
pred_set_includes_false <- eval_probs <= q_threshold

coverage <- mean(
  (as.logical(as.character(eval_data$success)) & pred_set_includes_true) |
  (!as.logical(as.character(eval_data$success)) & pred_set_includes_false)
)
cat(sprintf("Conformal coverage at alpha=%.2f: %.4f (target >= %.2f)\n",
            alpha, coverage, 1 - alpha))

# Flag high-uncertainty predictions (both classes included in prediction set)
uncertain_mask <- pred_set_includes_true & pred_set_includes_false
cat(sprintf("High-uncertainty predictions: %d / %d (%.1f%%)\n",
            sum(uncertain_mask), nrow(eval_data),
            100 * mean(uncertain_mask)))

eval_data$uncertain <- uncertain_mask
eval_data$pred_prob <- eval_probs

# ── Local Interpretability (LIME) ─────────────────────────────────────────────
# Select interesting cases: (1) most uncertain, (2) high-confidence success, 
# (3) high-confidence failure

uncertain_cases        <- eval_data[uncertain_mask, ] %>%
                            arrange(abs(pred_prob - 0.5)) %>%
                            head(3)
conf_success_cases     <- eval_data[!uncertain_mask, ] %>%
                            filter(success == "TRUE") %>%
                            arrange(desc(pred_prob)) %>%
                            head(2)
conf_fail_cases        <- eval_data[!uncertain_mask, ] %>%
                            filter(success == "FALSE") %>%
                            arrange(pred_prob) %>%
                            head(2)

lime_cases <- bind_rows(uncertain_cases, conf_success_cases, conf_fail_cases) %>%
  select(-uncertain, -pred_prob)

lime_explainer <- lime::lime(
  x       = train_data %>% select(-success),
  model   = rf_model,
  bin_continuous = TRUE
)

lime_explanations <- lime::explain(
  x              = lime_cases %>% select(-success),
  explainer      = lime_explainer,
  n_labels       = 1,
  n_features     = 5,
  n_permutations = 500
)

plot(lime::plot_features(lime_explanations)) +
  ggtitle("LIME: Local Feature Importance for Selected Cases")

# ── Failure Depth Analysis: Percent Climbed ───────────────────────────────────

df_failures <- merged_data %>%
  filter(success == FALSE | success == "FALSE") %>%
  filter(!is.na(highpoint_metres), !is.na(height_metres), height_metres > 0) %>%
  mutate(
    percent_climbed = pmin(highpoint_metres / height_metres, 1)
  ) %>%
  select(percent_climbed, age, sex, season, year, oxygen_used, hired, solo, height_metres)

df_failures_clean <- df_failures %>%
  filter(!is.na(age), !is.na(sex), sex != "Unknown", sex != "") %>%
  mutate(
    sex         = as.factor(sex),
    season      = as.factor(season),
    oxygen_used = as.factor(oxygen_used),
    hired       = as.factor(hired),
    solo        = as.factor(solo)
  ) %>%
  na.omit()

ggplot(df_failures_clean, aes(x = percent_climbed)) +
  geom_histogram(bins = 30, fill = "#4C72B0", color = "white") +
  scale_x_continuous(labels = percent_format()) +
  labs(title = "Distribution of % Climbed Among Failures",
       x = "Percentage of Mountain Climbed", y = "Count") +
  theme_minimal()

set.seed(123)
train_index_fail <- sample(seq_len(nrow(df_failures_clean)), size = 0.8 * nrow(df_failures_clean))
train_fail <- df_failures_clean[train_index_fail, ]
test_fail  <- df_failures_clean[-train_index_fail, ]

rf_fail_model <- randomForest(
  percent_climbed ~ .,
  data       = train_fail,
  ntree      = 500,
  importance = TRUE
)

pfun_reg <- function(object, newdata) predict(object, newdata = newdata)

X_explain_fail <- test_fail %>% select(-percent_climbed)

set.seed(42)
shap_values_fail <- fastshap::explain(
  rf_fail_model,
  X            = X_explain_fail,
  pred_wrapper = pfun_reg,
  nsim         = 50,
  adjust       = TRUE
)

shp_fail <- shapviz(shap_values_fail, X = X_explain_fail)

plot_fail_beeswarm <- sv_importance(shp_fail, kind = "beeswarm") +
  ggtitle("SHAP Summary: Predictors of Failure Depth (% Climbed)") +
  theme_minimal()

plot_fail_height <- sv_dependence(shp_fail, v = "height_metres", color_var = "oxygen_used") +
  ggtitle("SHAP Dependence: Peak Height × Oxygen Use (Failures)") +
  theme_minimal()

plot_fail_age <- sv_dependence(shp_fail, v = "age", color_var = "hired") +
  ggtitle("SHAP Dependence: Age × Hired Status (Failures)") +
  theme_minimal()

print(plot_fail_beeswarm)
print(plot_fail_height)
print(plot_fail_age)
