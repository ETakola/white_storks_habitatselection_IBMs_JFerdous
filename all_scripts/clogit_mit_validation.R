####Clogit mit validation####
# MATCHED used-available RSF with per-bird clogit models
# Purpose:
# - compare allowed population clogit models
# - choose best model by AIC
# - quantify intraspecific variability using per-bird clogit
# - export files for manuscript + IBM
#
# Matching unit:
# - fix_id = one used point + its matched available points
#####Load packages#####
library(tidyverse)
library(survival)
library(broom)

#####Settings#####
output_dir <- "[path for output directory]"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

dat <- readRDS("[path]/All_variables_RSF_scaled.rds")

# remove problematic sparse LULC classes
dat <- dat %>%
  filter(!LULC_class %in% c(20, 51, 61, 71, 130, 150, 181, 183))

min_sets_per_bird <- 20
min_used_per_bird <- 20
keep_lulc_in_per_bird <- FALSE

##### 1. Build common complete-case dataset #####
model_dat_all <- dat %>%
  select(
    fix_id, bird_id, used, LULC_class,
    ndvi, ndwi,
    ED_scaled, SHDI_scaled, SHEI_scaled, PD_scaled
  ) %>%
  drop_na() %>%
  mutate(
    bird_id = as.character(bird_id),
    fix_id = as.character(fix_id),
    LULC_class = as.factor(LULC_class),
    used = as.integer(used)
  )

cat("Rows in model_dat_all:", nrow(model_dat_all), "\n")
cat("Choice sets in model_dat_all:", n_distinct(model_dat_all$fix_id), "\n")
cat("Birds in model_dat_all:", n_distinct(model_dat_all$bird_id), "\n")


#####2. Verify matched structure####

matching_check <- model_dat_all %>%
  group_by(fix_id) %>%
  summarise(
    n_rows = n(),
    n_used = sum(used == 1, na.rm = TRUE),
    n_available = sum(used == 0, na.rm = TRUE),
    n_birds = n_distinct(bird_id),
    .groups = "drop"
  )

write_csv(matching_check, file.path(output_dir, "matching_structure_check.csv"))

cat("\nDistribution of n_used per fix_id:\n")
print(table(matching_check$n_used))

cat("\nDistribution of n_birds per fix_id:\n")
print(table(matching_check$n_birds))

bad_sets <- matching_check %>%
  filter(n_used != 1 | n_birds != 1 | n_available < 1)

cat("\nNumber of invalid fix_id sets:", nrow(bad_sets), "\n")

if (nrow(bad_sets) > 0) {
  warning("Invalid matching structure detected. Check matching_structure_check.csv")
}

library(dplyr)
library(readr)
library(car)


######RSF REPORTING NUMBERS######
# 1) valid matched choice sets only
valid_fix_ids <- matching_check %>%
  filter(n_used == 1, n_birds == 1, n_available >= 1) %>%
  pull(fix_id)

valid_dat <- model_dat_all %>%
  filter(fix_id %in% valid_fix_ids)

# 2) basic matched-set numbers
rsf_dataset_summary <- tibble(
  n_valid_choice_sets = n_distinct(valid_dat$fix_id),
  n_used_points = sum(valid_dat$used == 1),
  n_available_points = sum(valid_dat$used == 0),
  used_available_ratio = paste0("1:", round(sum(valid_dat$used == 0) / sum(valid_dat$used == 1), 2)),
  n_birds = n_distinct(valid_dat$bird_id)
)

print(rsf_dataset_summary)

write_csv(
  rsf_dataset_summary,
  file.path(output_dir, "RSF_dataset_summary_for_text.csv")
)

# 3) retained LULC classes
retained_lulc <- valid_dat %>%
  distinct(LULC_class) %>%
  arrange(LULC_class)

print(retained_lulc)

write_csv(
  retained_lulc,
  file.path(output_dir, "retained_LULC_classes.csv")
)



###### Multicolinearity check#######
vif_mod <- glm(
  used ~ ndwi + ED_scaled + LULC_class,
  data = valid_dat,
  family = binomial()
)

vif_values <- car::vif(vif_mod)
print(vif_values)

capture.output(
  vif_values,
  file = file.path(output_dir, "RSF_multicollinearity_VIF_ndwi.txt")
)

vif_mod1 <- glm(
  used ~ ndvi + ED_scaled + LULC_class,
  data = valid_dat,
  family = binomial()
)

vif_values1 <- car::vif(vif_mod1)
print(vif_values1)

capture.output(
  vif_values1,
  file = file.path(output_dir, "RSF_multicollinearity_VIF_ndvi.txt")
)

# Numeric predictor correlation
numeric_cor <- valid_dat %>%
  select(ndvi, ndwi, ED_scaled, SHDI_scaled, SHEI_scaled, PD_scaled) %>%
  cor(use = "complete.obs")

print(numeric_cor)

write.csv(
  numeric_cor,
  file.path(output_dir, "RSF_numeric_predictor_correlation_matrix.csv")
)
##### 3. Define allowed predictor combinations #####

veg_vars <- c("ndvi", "ndwi")
land_vars <- c("ED_scaled", "SHDI_scaled", "SHEI_scaled", "PD_scaled")

model_grid <- tidyr::crossing(
  veg = veg_vars,
  land = land_vars
) %>%
  mutate(model_name = paste0("m_", veg, "_", land))

print(model_grid)


##### 4. Fit population clogit models #####
fit_one_population_model <- function(veg, land, model_name, data) {
  form_txt <- paste0(
    "used ~ ", veg, " + ", land, " + LULC_class + strata(fix_id) + cluster(bird_id)"
  )
  
  form <- as.formula(form_txt)
  
  fit <- clogit(
    formula = form,
    data = data,
    method = "efron"
  )
  
  list(
    name = model_name,
    veg = veg,
    land = land,
    formula = form,
    fit = fit
  )
}

all_models <- purrr::pmap(
  list(model_grid$veg, model_grid$land, model_grid$model_name),
  ~ fit_one_population_model(..1, ..2, ..3, data = model_dat_all)
)

names(all_models) <- model_grid$model_name


##### 5. Compare AIC #####
aic_table <- purrr::map_dfr(all_models, function(x) {
  tibble(
    model = x$name,
    veg = x$veg,
    land = x$land,
    formula = deparse(x$formula),
    AIC = AIC(x$fit)
  )
}) %>%
  arrange(AIC)

print(aic_table, n = Inf)
write_csv(aic_table, file.path(output_dir, "clogit_model_AIC_comparison.csv"))


##### 6. Select and export best population model#####
##We checked the best model here##
##But selected ndvi model later in the code for ecological reasons##
best_model_name <- aic_table$model[1]
best_model_obj <- all_models[[best_model_name]]
best_model <- best_model_obj$fit
best_veg <- best_model_obj$veg
best_land <- best_model_obj$land

cat("\nBest model by AIC:", best_model_name, "\n")
cat("Best vegetation predictor:", best_veg, "\n")
cat("Best landscape predictor:", best_land, "\n")
print(summary(best_model))

best_sum <- summary(best_model)

population_fixed_effects <- tibble(
  term = rownames(best_sum$coefficients),
  beta = best_sum$coefficients[, "coef"],
  se = best_sum$coefficients[, "se(coef)"],
  z = best_sum$coefficients[, "z"],
  p = best_sum$coefficients[, "Pr(>|z|)"],
  exp_beta = best_sum$coefficients[, "exp(coef)"]
)

write_csv(
  population_fixed_effects,
  file.path(output_dir, "population_best_model_fixed_effects.csv")
)

# IBM-ready fixed effects
ibm_fixed_effects <- population_fixed_effects %>%
  select(term, beta) %>%
  rename(coef = beta)

write_csv(
  ibm_fixed_effects,
  file.path(output_dir, "IBM_fixed_effects_best_model.csv")
)


##### 7. Build per-bird dataset for best model#####

per_bird_dat <- model_dat_all %>%
  select(
    fix_id, bird_id, used, LULC_class,
    all_of(best_veg),
    all_of(best_land)
  )

bird_counts <- per_bird_dat %>%
  group_by(bird_id) %>%
  summarise(
    n_rows = n(),
    n_sets = n_distinct(fix_id),
    n_used = sum(used == 1, na.rm = TRUE),
    n_available = sum(used == 0, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(n_sets))

write_csv(
  bird_counts,
  file.path(output_dir, "per_bird_data_counts_best_model.csv")
)

eligible_birds <- bird_counts %>%
  filter(n_sets >= min_sets_per_bird, n_used >= min_used_per_bird) %>%
  pull(bird_id)

cat("\nEligible birds for per-bird models:", length(eligible_birds), "\n")
print(eligible_birds)


##### 8. Fit per-bird clogit models#####

fit_one_bird_model <- function(bid, data, veg, land, keep_lulc = FALSE) {
  dsub <- data %>%
    filter(bird_id == bid) %>%
    droplevels()
  
  n_sets <- n_distinct(dsub$fix_id)
  n_used <- sum(dsub$used == 1, na.rm = TRUE)
  n_rows <- nrow(dsub)
  
  if (n_sets < min_sets_per_bird || n_used < min_used_per_bird) {
    return(list(
      bird_id = bid,
      ok = FALSE,
      reason = "insufficient_data",
      n_rows = n_rows,
      n_sets = n_sets,
      n_used = n_used,
      fit = NULL
    ))
  }
  
  if (keep_lulc) {
    form_txt <- paste0(
      "used ~ ", veg, " + ", land, " + LULC_class + strata(fix_id)"
    )
  } else {
    form_txt <- paste0(
      "used ~ ", veg, " + ", land, " + strata(fix_id)"
    )
  }
  
  form <- as.formula(form_txt)
  
  fit_try <- try(
    clogit(
      formula = form,
      data = dsub,
      method = "efron"
    ),
    silent = TRUE
  )
  
  if (inherits(fit_try, "try-error")) {
    return(list(
      bird_id = bid,
      ok = FALSE,
      reason = "fit_failed",
      n_rows = n_rows,
      n_sets = n_sets,
      n_used = n_used,
      fit = NULL
    ))
  }
  
  list(
    bird_id = bid,
    ok = TRUE,
    reason = "ok",
    n_rows = n_rows,
    n_sets = n_sets,
    n_used = n_used,
    fit = fit_try
  )
}

bird_model_results <- purrr::map(
  eligible_birds,
  ~ fit_one_bird_model(
    bid = .x,
    data = per_bird_dat,
    veg = best_veg,
    land = best_land,
    keep_lulc = keep_lulc_in_per_bird
  )
)

names(bird_model_results) <- eligible_birds


##### 9. Export per-bird diagnostics #####

per_bird_diagnostics <- purrr::map_dfr(bird_model_results, function(x) {
  if (!x$ok) {
    tibble(
      bird_id = x$bird_id,
      ok = FALSE,
      reason = x$reason,
      n_rows = x$n_rows,
      n_sets = x$n_sets,
      n_used = x$n_used,
      AIC = NA_real_,
      logLik = NA_real_
    )
  } else {
    tibble(
      bird_id = x$bird_id,
      ok = TRUE,
      reason = x$reason,
      n_rows = x$n_rows,
      n_sets = x$n_sets,
      n_used = x$n_used,
      AIC = AIC(x$fit),
      logLik = as.numeric(logLik(x$fit))
    )
  }
})

write_csv(
  per_bird_diagnostics,
  file.path(output_dir, "per_bird_model_diagnostics_best_model.csv")
)


##### 10. Export per-bird coefficients#####

extract_one_bird_coefficients <- function(res) {
  if (!res$ok) return(NULL)
  
  sm <- summary(res$fit)
  
  tibble(
    bird_id = res$bird_id,
    term = rownames(sm$coefficients),
    beta = sm$coefficients[, "coef"],
    se = sm$coefficients[, "se(coef)"],
    z = sm$coefficients[, "z"],
    p = sm$coefficients[, "Pr(>|z|)"],
    exp_beta = sm$coefficients[, "exp(coef)"]
  )
}

per_bird_coefficients_long <- purrr::map_dfr(
  bird_model_results,
  extract_one_bird_coefficients
)

write_csv(
  per_bird_coefficients_long,
  file.path(output_dir, "per_bird_coefficients_best_model_long.csv")
)

per_bird_coefficients_wide <- per_bird_coefficients_long %>%
  select(bird_id, term, beta) %>%
  pivot_wider(names_from = term, values_from = beta) %>%
  arrange(bird_id)

write_csv(
  per_bird_coefficients_wide,
  file.path(output_dir, "per_bird_coefficients_best_model.csv")
)

# IBM-ready per-bird slopes
ibm_per_bird <- per_bird_coefficients_wide
write_csv(
  ibm_per_bird,
  file.path(output_dir, "IBM_bird_random_effects_best_model.csv")
)


##### 11. Summarise intraspecific variability#####

target_terms <- c(best_veg, best_land)

intraspecific_variability_summary <- per_bird_coefficients_long %>%
  filter(term %in% target_terms) %>%
  group_by(term) %>%
  summarise(
    n_birds = n(),
    mean_beta = mean(beta, na.rm = TRUE),
    sd_beta = sd(beta, na.rm = TRUE),
    min_beta = min(beta, na.rm = TRUE),
    q25_beta = quantile(beta, 0.25, na.rm = TRUE),
    median_beta = median(beta, na.rm = TRUE),
    q75_beta = quantile(beta, 0.75, na.rm = TRUE),
    max_beta = max(beta, na.rm = TRUE),
    n_positive = sum(beta > 0, na.rm = TRUE),
    n_negative = sum(beta < 0, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  intraspecific_variability_summary,
  file.path(output_dir, "intraspecific_variability_summary_best_model.csv")
)


##### 12. Save all population summaries#####
sink(file.path(output_dir, "all_population_model_summaries.txt"))
for (nm in names(all_models)) {
  cat("\n====================================================\n")
  cat("MODEL:", nm, "\n")
  cat("====================================================\n")
  print(summary(all_models[[nm]]$fit))
}
sink()


##### 13. Final reporting#####
cat("\n====================================================\n")
cat("DONE\n")
cat("====================================================\n")
cat("Best population model:", best_model_name, "\n")
cat("Best vegetation predictor:", best_veg, "\n")
cat("Best landscape predictor:", best_land, "\n")
cat("Eligible birds for per-bird models:", length(eligible_birds), "\n")
cat("Successful per-bird models:", sum(per_bird_diagnostics$ok), "\n")
cat("\nFiles written to:\n", output_dir, "\n")



###### FIT SPECIFIC MODEL (m_ndvi_ED_scaled)######
model_formula <- used ~ ndvi + ED_scaled + LULC_class + strata(fix_id) + cluster(bird_id)

model_m1 <- clogit(
  formula = model_formula,
  data = model_dat_all,
  method = "efron"
)

summary(model_m1)


###### EXPORT POPULATION FIXED EFFECTS (for IBM)######

coef_df <- summary(model_m1)$coefficients

population_fixed_effects <- tibble(
  term = rownames(coef_df),
  coef = coef_df[, "coef"]
)

write_csv(population_fixed_effects,
          file.path(output_dir, "IBM_fixed_effects_ndvi_model.csv"))



###### PER-BIRD MODELS (INTRASPECIFIC VARIATION)#####

fit_one_bird <- function(bid) {
  dsub <- model_dat_all %>% filter(bird_id == bid)
  
  if (n_distinct(dsub$fix_id) < 20) return(NULL)
  
  fit <- try(
    clogit(
      used ~ ndvi + ED_scaled + strata(fix_id),
      data = dsub,
      method = "efron"
    ),
    silent = TRUE
  )
  
  if (inherits(fit, "try-error")) return(NULL)
  
  coefs <- summary(fit)$coefficients
  
  tibble(
    bird_id = bid,
    ndvi = coefs["ndvi", "coef"],
    ED_scaled = coefs["ED_scaled", "coef"]
  )
}

bird_ids <- unique(model_dat_all$bird_id)

per_bird_coefs <- purrr::map_dfr(bird_ids, fit_one_bird)

write_csv(per_bird_coefs,
          file.path(output_dir, "IBM_bird_specific_slopes_ndvi.csv"))



#### QUICK VARIABILITY SUMMARY####

variability_summary <- per_bird_coefs %>%
  summarise(
    ndvi_mean = mean(ndvi, na.rm = TRUE),
    ndvi_sd = sd(ndvi, na.rm = TRUE),
    ED_mean = mean(ED_scaled, na.rm = TRUE),
    ED_sd = sd(ED_scaled, na.rm = TRUE)
  )

write_csv(variability_summary,
          file.path(output_dir, "intraspecific_variability_summary_ndvi.csv"))

# load both
fixed <- read_csv(file.path(output_dir, "IBM_fixed_effects_ndvi_model.csv"))
bird  <- read_csv(file.path(output_dir, "IBM_bird_specific_slopes_ndvi.csv"))

# extract population values
beta_ndvi <- fixed %>% filter(term == "ndvi") %>% pull(coef)
beta_ed   <- fixed %>% filter(term == "ED_scaled") %>% pull(coef)

# convert to deviations
bird_random <- bird %>%
  mutate(
    re_ndvi = ndvi - beta_ndvi,
    re_ED_scaled = ED_scaled - beta_ed
  ) %>%
  select(bird_id, re_ndvi, re_ED_scaled)

write_csv(bird_random,
          file.path(output_dir, "IBM_bird_random_effects_ndvi_model.csv"))

#####LOBO:Leave-One-Bird-Out Cross Validation#####
library(tidyverse)
library(survival)


##### manual clogit scoring without strata#####

score_clogit_manual <- function(fit, newdata) {
  X <- model.matrix(~ ndvi + ED_scaled + LULC_class, data = newdata)
  beta <- coef(fit)
  
  # keep only columns that exist in fitted coefficients
  common_cols <- intersect(colnames(X), names(beta))
  
  as.numeric(X[, common_cols, drop = FALSE] %*% beta[common_cols])
}


###### leave-one-bird-out validation#####

library(tidyverse)
library(survival)

score_clogit_manual <- function(fit, newdata) {
  X <- model.matrix(~ ndvi + ED_scaled + LULC_class, data = newdata)
  beta <- coef(fit)
  
  common_cols <- intersect(colnames(X), names(beta))
  
  as.numeric(X[, common_cols, drop = FALSE] %*% beta[common_cols])
}

birds <- eligible_birds

cv_results <- map_dfr(birds, function(test_bird) {
  
  train <- model_dat_all %>%
    filter(bird_id != test_bird) %>%
    mutate(LULC_class = droplevels(LULC_class))
  
  test <- model_dat_all %>%
    filter(bird_id == test_bird) %>%
    mutate(LULC_class = factor(LULC_class, levels = levels(train$LULC_class)))
  
  fit <- clogit(
    used ~ ndvi + ED_scaled + LULC_class + strata(fix_id),
    data = train,
    method = "efron"
  )
  
  test <- test %>%
    mutate(score = score_clogit_manual(fit, .))
  
  test %>%
    group_by(fix_id) %>%
    arrange(desc(score), .by_group = TRUE) %>%
    mutate(rank = row_number()) %>%
    summarise(
      used_rank = rank[used == 1][1],
      top1 = as.integer(used_rank == 1),
      top3 = as.integer(used_rank <= 3),
      .groups = "drop"
    ) %>%
    summarise(
      bird_id = test_bird,
      n_sets = n(),
      top1_acc = mean(top1, na.rm = TRUE),
      top3_acc = mean(top3, na.rm = TRUE),
      mean_rank = mean(used_rank, na.rm = TRUE)
    )
})

print(cv_results)

cv_summary <- cv_results %>%
  summarise(
    n_birds = n(),
    mean_top1_acc = mean(top1_acc, na.rm = TRUE),
    sd_top1_acc = sd(top1_acc, na.rm = TRUE),
    min_top1_acc = min(top1_acc, na.rm = TRUE),
    max_top1_acc = max(top1_acc, na.rm = TRUE),
    mean_top3_acc = mean(top3_acc, na.rm = TRUE),
    sd_top3_acc = sd(top3_acc, na.rm = TRUE),
    mean_rank = mean(mean_rank, na.rm = TRUE),
    sd_rank = sd(mean_rank, na.rm = TRUE)
  )

print(cv_summary)
model_dat_all %>%
  count(fix_id) %>%
  summarise(
    mean_options = mean(n),
    median_options = median(n),
    min_options = min(n),
    max_options = max(n)
  )

write_csv(cv_results, file.path(output_dir, "LOBO_validation_per_bird.csv"))
write_csv(cv_summary, file.path(output_dir, "LOBO_validation_summary.csv"))

#####Full Model Ranking Accuracy#####
model_dat_all$score <- predict(model_m1, type="lp")

rank_results <- model_dat_all %>%
  group_by(fix_id) %>%
  arrange(desc(score), .by_group=TRUE) %>%
  mutate(rank = row_number()) %>%
  summarise(
    used_rank = rank[used == 1][1]
  ) %>%
  summarise(
    top1 = mean(used_rank == 1),
    top3 = mean(used_rank <= 3),
    mean_rank = mean(used_rank)
  )
rank_results
write_csv(rank_results, file.path(output_dir, "ranking_validation.csv"))

#####Bootstrap Stability#####
set.seed(123)

B <- 200
boot_out <- map_dfr(1:B, function(i){
  
  samp_birds <- sample(unique(model_dat_all$bird_id),
                       replace = TRUE)
  
  dboot <- bind_rows(lapply(samp_birds, function(b){
    model_dat_all %>% filter(bird_id == b)
  }))
  
  fit <- try(
    clogit(
      used ~ ndvi + ED_scaled + LULC_class + strata(fix_id),
      data = dboot
    ),
    silent = TRUE
  )
  
  if(inherits(fit, "try-error")) return(NULL)
  
  tibble(
    iter = i,
    ndvi = coef(fit)["ndvi"],
    ED_scaled = coef(fit)["ED_scaled"]
  )
})

boot_summary <- boot_out %>%
  summarise(
    ndvi_mean = mean(ndvi, na.rm=TRUE),
    ndvi_lwr = quantile(ndvi, 0.025, na.rm=TRUE),
    ndvi_upr = quantile(ndvi, 0.975, na.rm=TRUE),
    
    ed_mean = mean(ED_scaled, na.rm=TRUE),
    ed_lwr = quantile(ED_scaled, 0.025, na.rm=TRUE),
    ed_upr = quantile(ED_scaled, 0.975, na.rm=TRUE)
  )
boot_summary
write_csv(boot_summary, file.path(output_dir, "Boot_validation_summary.csv"))
#####blocked k-fold by bird#####
library(tidyverse)
library(survival)
library(rsample)


###### SETTINGS ######

set.seed(123)

k_folds <- 5
cv_dat <- model_dat_all %>%
  select(fix_id, bird_id, used, LULC_class, ndvi, ED_scaled) %>%
  mutate(
    bird_id = as.character(bird_id),
    fix_id = as.character(fix_id),
    used = as.integer(used),
    LULC_class = as.factor(LULC_class)
  )


eligible_birds <- cv_dat %>%
  group_by(bird_id) %>%
  summarise(
    n_sets = n_distinct(fix_id),
    n_used = sum(used == 1),
    .groups = "drop"
  ) %>%
  filter(n_sets >= 20, n_used >= 20) %>%
  pull(bird_id)

cv_dat <- cv_dat %>%
  filter(bird_id %in% eligible_birds) %>%
  droplevels()

cat("Birds used in blocked k-fold:", n_distinct(cv_dat$bird_id), "\n")
cat("Choice sets used in blocked k-fold:", n_distinct(cv_dat$fix_id), "\n")


###### CREATE BLOCKED FOLDS BY BIRD######

bird_df <- tibble(bird_id = sort(unique(cv_dat$bird_id)))

folds <- group_vfold_cv(
  bird_df,
  group = bird_id,
  v = k_folds
)


######Evaluate ranking within strata#####

evaluate_stratum_ranking <- function(test_data, score_col = "score") {
  test_data %>%
    group_by(fix_id) %>%
    arrange(desc(.data[[score_col]]), .by_group = TRUE) %>%
    mutate(rank = row_number()) %>%
    summarise(
      used_rank = rank[used == 1][1],
      n_options = n(),
      .groups = "drop"
    ) %>%
    mutate(
      top1 = as.integer(used_rank == 1),
      top3 = as.integer(used_rank <= 3)
    )
}


###### RUN BLOCKED K-FOLD CV ######

library(tidyverse)
library(survival)
library(rsample)

score_clogit_manual <- function(fit, newdata) {
  X <- model.matrix(~ ndvi + ED_scaled + LULC_class, data = newdata)
  beta <- coef(fit)
  common_cols <- intersect(colnames(X), names(beta))
  as.numeric(X[, common_cols, drop = FALSE] %*% beta[common_cols])
}

evaluate_stratum_ranking <- function(test_data) {
  test_data %>%
    group_by(fix_id) %>%
    arrange(desc(score), .by_group = TRUE) %>%
    mutate(rank = row_number()) %>%
    summarise(
      used_rank = rank[used == 1][1],
      n_options = n(),
      .groups = "drop"
    ) %>%
    mutate(
      top1 = as.integer(used_rank == 1),
      top3 = as.integer(used_rank <= 3)
    )
}

set.seed(123)
k_folds <- 5

cv_dat <- model_dat_all %>%
  filter(bird_id %in% eligible_birds) %>%
  mutate(
    bird_id = as.character(bird_id),
    fix_id = as.character(fix_id),
    used = as.integer(used),
    LULC_class = as.factor(LULC_class)
  )

bird_df <- tibble(bird_id = sort(unique(cv_dat$bird_id)))

folds <- rsample::group_vfold_cv(
  bird_df,
  group = bird_id,
  v = k_folds
)

print(folds)

fold_results <- list()

for (i in seq_len(nrow(folds))) {
  cat("\n============================\n")
  cat("Fold", i, "\n")
  cat("============================\n")
  
  split_obj <- folds$splits[[i]]
  
  train_birds <- rsample::analysis(split_obj) %>% pull(bird_id)
  test_birds  <- rsample::assessment(split_obj) %>% pull(bird_id)
  
  cat("Train birds:", length(train_birds), "\n")
  cat("Test birds:", length(test_birds), "\n")
  
  train_dat <- cv_dat %>% filter(bird_id %in% train_birds)
  test_dat  <- cv_dat %>% filter(bird_id %in% test_birds)
  
  cat("Train rows:", nrow(train_dat), "\n")
  cat("Test rows:", nrow(test_dat), "\n")
  cat("Train sets:", n_distinct(train_dat$fix_id), "\n")
  cat("Test sets:", n_distinct(test_dat$fix_id), "\n")
  
  if (nrow(train_dat) == 0 || nrow(test_dat) == 0) {
    cat("Skipping fold because train or test is empty\n")
    next
  }
  
  train_dat <- train_dat %>% mutate(LULC_class = droplevels(LULC_class))
  test_dat <- test_dat %>%
    mutate(LULC_class = factor(LULC_class, levels = levels(train_dat$LULC_class)))
  
  fit <- try(
    clogit(
      used ~ ndvi + ED_scaled + LULC_class + strata(fix_id),
      data = train_dat,
      method = "efron"
    ),
    silent = TRUE
  )
  
  if (inherits(fit, "try-error")) {
    cat("Model fit failed in fold", i, "\n")
    print(fit)
    next
  }
  
  test_dat <- test_dat %>%
    mutate(score = score_clogit_manual(fit, .))
  
  stratum_eval <- evaluate_stratum_ranking(test_dat)
  
  fold_summary <- stratum_eval %>%
    summarise(
      fold = i,
      n_test_birds = n_distinct(test_dat$bird_id),
      n_test_sets = n(),
      top1_acc = mean(top1, na.rm = TRUE),
      top3_acc = mean(top3, na.rm = TRUE),
      mean_rank = mean(used_rank, na.rm = TRUE),
      median_rank = median(used_rank, na.rm = TRUE)
    )
  
  print(fold_summary)
  
  fold_results[[length(fold_results) + 1]] <- list(
    fold_summary = fold_summary,
    stratum_eval = stratum_eval
  )
}

cv_fold_summary <- bind_rows(purrr::map(fold_results, "fold_summary"))
print(cv_fold_summary)

# PRINT
cat("\n====================================================\n")
cat("BLOCKED K-FOLD CV DONE\n")
cat("====================================================\n")

cat("\nFold-wise summary:\n")
print(cv_fold_summary)

cv_overall_summary <- cv_fold_summary %>%
  summarise(
    k_folds = n(),
    mean_top1_acc = mean(top1_acc, na.rm = TRUE),
    sd_top1_acc = sd(top1_acc, na.rm = TRUE),
    min_top1_acc = min(top1_acc, na.rm = TRUE),
    max_top1_acc = max(top1_acc, na.rm = TRUE),
    mean_top3_acc = mean(top3_acc, na.rm = TRUE),
    sd_top3_acc = sd(top3_acc, na.rm = TRUE),
    min_top3_acc = min(top3_acc, na.rm = TRUE),
    max_top3_acc = max(top3_acc, na.rm = TRUE),
    mean_mean_rank = mean(mean_rank, na.rm = TRUE),
    sd_mean_rank = sd(mean_rank, na.rm = TRUE)
  )

print(cv_overall_summary)
write_csv(cv_fold_summary, file.path(output_dir, "blocked_kfold_by_bird_fold_summary.csv"))
write_csv(cv_overall_summary, file.path(output_dir, "blocked_kfold_by_bird_overall_summary.csv"))
cat("\nFiles written to:\n", output_dir, "\n")

###### FOR TEXT: BIRD-SPECIFIC VARIATION SUMMARY FOR TEXT#####
# per_bird_coefs should contain: bird_id, ndvi, ED_scaled
# population coefficients
pop_ndvi <- coef(model_m1)["ndvi"]
pop_ed   <- coef(model_m1)["ED_scaled"]

bird_variation_summary <- per_bird_coefs %>%
  summarise(
    n_birds = n(),
    
    ndvi_min = min(ndvi, na.rm = TRUE),
    ndvi_q25 = quantile(ndvi, 0.25, na.rm = TRUE),
    ndvi_median = median(ndvi, na.rm = TRUE),
    ndvi_q75 = quantile(ndvi, 0.75, na.rm = TRUE),
    ndvi_max = max(ndvi, na.rm = TRUE),
    ndvi_sd = sd(ndvi, na.rm = TRUE),
    
    ed_min = min(ED_scaled, na.rm = TRUE),
    ed_q25 = quantile(ED_scaled, 0.25, na.rm = TRUE),
    ed_median = median(ED_scaled, na.rm = TRUE),
    ed_q75 = quantile(ED_scaled, 0.75, na.rm = TRUE),
    ed_max = max(ED_scaled, na.rm = TRUE),
    ed_sd = sd(ED_scaled, na.rm = TRUE)
  )

print(bird_variation_summary)

write_csv(
  bird_variation_summary,
  file.path(output_dir, "bird_specific_variation_summary_for_text.csv")
)

###### DIRECTIONAL AGREEMENT WITH POPULATION EFFECT######


direction_agreement <- per_bird_coefs %>%
  mutate(
    ndvi_same_direction = sign(ndvi) == sign(pop_ndvi),
    ed_same_direction = sign(ED_scaled) == sign(pop_ed),
    
    ndvi_contrasting = sign(ndvi) != sign(pop_ndvi),
    ed_contrasting = sign(ED_scaled) != sign(pop_ed)
  ) %>%
  summarise(
    n_birds = n(),
    
    pop_ndvi_beta = pop_ndvi,
    ndvi_same_n = sum(ndvi_same_direction, na.rm = TRUE),
    ndvi_same_pct = 100 * mean(ndvi_same_direction, na.rm = TRUE),
    ndvi_contrast_n = sum(ndvi_contrasting, na.rm = TRUE),
    ndvi_contrast_pct = 100 * mean(ndvi_contrasting, na.rm = TRUE),
    
    pop_ed_beta = pop_ed,
    ed_same_n = sum(ed_same_direction, na.rm = TRUE),
    ed_same_pct = 100 * mean(ed_same_direction, na.rm = TRUE),
    ed_contrast_n = sum(ed_contrasting, na.rm = TRUE),
    ed_contrast_pct = 100 * mean(ed_contrasting, na.rm = TRUE)
  )

print(direction_agreement)

write_csv(
  direction_agreement,
  file.path(output_dir, "bird_direction_agreement_with_population.csv")
)

###### SIMPLE HETEROGENEITY CLASSIFICATION######
# Based on coefficient sign reversal and spread


heterogeneity_class <- per_bird_coefs %>%
  summarise(
    ndvi_sign_reversal_pct = 100 * mean(sign(ndvi) != sign(pop_ndvi), na.rm = TRUE),
    ed_sign_reversal_pct = 100 * mean(sign(ED_scaled) != sign(pop_ed), na.rm = TRUE),
    
    ndvi_range = max(ndvi, na.rm = TRUE) - min(ndvi, na.rm = TRUE),
    ed_range = max(ED_scaled, na.rm = TRUE) - min(ED_scaled, na.rm = TRUE)
  ) %>%
  mutate(
    heterogeneity = case_when(
      ndvi_sign_reversal_pct < 10 & ed_sign_reversal_pct < 10 ~ "weak",
      ndvi_sign_reversal_pct < 30 & ed_sign_reversal_pct < 30 ~ "moderate",
      TRUE ~ "substantial"
    )
  )

print(heterogeneity_class)

write_csv(
  heterogeneity_class,
  file.path(output_dir, "heterogeneity_classification_for_text.csv")
)


###### FOR TEXT: VALIDATION SUMMARY #####
validation_summary_for_text <- tibble(
  method = c(
    "LOBO cross-validation",
    "Bird-blocked 5-fold cross-validation",
    "In-sample ranking"
  ),
  
  mean_rank = c(
    cv_summary$mean_rank,
    cv_overall_summary$mean_mean_rank,
    rank_results$mean_rank
  ),
  
  top1_percent = c(
    cv_summary$mean_top1_acc * 100,
    cv_overall_summary$mean_top1_acc * 100,
    rank_results$top1 * 100
  ),
  
  top3_percent = c(
    cv_summary$mean_top3_acc * 100,
    cv_overall_summary$mean_top3_acc * 100,
    rank_results$top3 * 100
  )
)

print(validation_summary_for_text)

write_csv(
  validation_summary_for_text,
  file.path(output_dir, "RSF_validation_summary_for_text.csv")
)

###### FOR TEXT: BOOTSTRAP CI SUMMARY  ######

bootstrap_summary_for_text <- boot_summary %>%
  mutate(
    ndvi_stability = case_when(
      ndvi_lwr > 0 & ndvi_upr > 0 ~ "stable positive",
      ndvi_lwr < 0 & ndvi_upr < 0 ~ "stable negative",
      TRUE ~ "variable / uncertain"
    ),
    ed_stability = case_when(
      ed_lwr > 0 & ed_upr > 0 ~ "stable positive",
      ed_lwr < 0 & ed_upr < 0 ~ "stable negative",
      TRUE ~ "variable / uncertain"
    )
  )

print(bootstrap_summary_for_text)

write_csv(
  bootstrap_summary_for_text,
  file.path(output_dir, "RSF_bootstrap_summary_for_text.csv")
)
