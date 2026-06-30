###Jannatul
###Modified: 21/05/2026
library(tidyverse)
library(survival)
library(ggplot2)

output_dir <- "[path]/RSF_intraspecific_outputs"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

dat <- readRDS("[path]/All_variables_RSF_scaled.rds")
dat <- dat %>%
  filter(!LULC_class %in% c(20, 51, 61, 71, 130, 150, 181, 183))
min_sets_per_bird <- 20
min_used_per_bird <- 20


veg <- "ndvi"
land <- "ED_scaled"   


#### DATA ####

model_dat_all <- dat %>%
  select(
    fix_id, bird_id, used, LULC_class,
    ndvi, ndwi,
    ED_scaled, SHDI_scaled, SHEI_scaled, PD_scaled
  ) %>%
  drop_na() %>%
  mutate(
    set_id = as.character(fix_id),
    bird_id = as.character(bird_id),
    used = as.integer(used),
    LULC_class = as.factor(LULC_class)
  )

per_bird_dat <- model_dat_all %>%
  select(set_id, bird_id, used, LULC_class, all_of(veg), all_of(land))

bird_counts <- per_bird_dat %>%
  group_by(bird_id) %>%
  summarise(
    n_rows = n(),
    n_sets = n_distinct(set_id),
    n_used = sum(used == 1, na.rm = TRUE),
    n_available = sum(used == 0, na.rm = TRUE),
    .groups = "drop"
  )

eligible_birds <- bird_counts %>%
  filter(n_sets >= min_sets_per_bird, n_used >= min_used_per_bird) %>%
  pull(bird_id)

cat("Eligible birds:", length(eligible_birds), "\n")


#### FUNCTIONS ####

fit_one_bird_model <- function(bid, data, veg, land) {
  dsub <- data %>%
    filter(bird_id == bid) %>%
    droplevels()
  
  n_sets <- n_distinct(dsub$set_id)
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
  
  form_txt <- paste0(
    "used ~ ", veg, " + ", land, " + LULC_class + strata(set_id)"
  )
  
  fit_try <- try(
    clogit(
      formula = as.formula(form_txt),
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


#### FIT PER-BIRD MODELS WITH LULC ####

bird_model_results <- purrr::map(
  eligible_birds,
  ~ fit_one_bird_model(
    bid = .x,
    data = per_bird_dat,
    veg = veg,
    land = land
  )
)

names(bird_model_results) <- eligible_birds

per_bird_diagnostics <- purrr::map_dfr(bird_model_results, function(x) {
  tibble(
    bird_id = x$bird_id,
    ok = x$ok,
    reason = x$reason,
    n_rows = x$n_rows,
    n_sets = x$n_sets,
    n_used = x$n_used,
    AIC = ifelse(x$ok, AIC(x$fit), NA_real_),
    logLik = ifelse(x$ok, as.numeric(logLik(x$fit)), NA_real_)
  )
})

write_csv(
  per_bird_diagnostics,
  file.path(output_dir, "per_bird_model_diagnostics_NDVI_ED_with_LULC.csv")
)

print(per_bird_diagnostics)

cat("Successful bird models:", sum(per_bird_diagnostics$ok), "\n")

#### EXTRACT COEFFICIENTS ####
coef_long <- purrr::map_dfr(
  bird_model_results,
  extract_one_bird_coefficients
)

write_csv(
  coef_long,
  file.path(output_dir, "per_bird_coefficients_NDVI_ED_with_LULC_long.csv")
)


#### PLOT DATA: class-wise LULC violins ####

plot_df <- coef_long %>%
  mutate(
    term_clean = case_when(
      term == "ndvi" ~ "NDVI",
      term == "ndwi" ~ "NDWI",
      term == "ED_scaled" ~ "Edge density",
      term == "SHDI_scaled" ~ "SHDI",
      term == "SHEI_scaled" ~ "SHEI",
      term == "PD_scaled" ~ "Patch density",
      str_starts(term, "LULC_class") ~ term,
      TRUE ~ term
    ),
    bird_id = as.factor(bird_id)
  ) %>%
  filter(is.finite(beta)) %>%
  filter(abs(beta) < 10)


 
# Order predictors

lulc_order <- plot_df %>%
  filter(str_starts(term_clean, "LULC_class")) %>%
  mutate(lulc_num = as.numeric(str_replace(term_clean, "LULC_class", ""))) %>%
  arrange(lulc_num) %>%
  pull(term_clean) %>%
  unique()

main_order <- c("NDVI", "NDWI", "Edge density", "SHDI", "SHEI", "Patch density")
main_order <- main_order[main_order %in% plot_df$term_clean]

plot_df <- plot_df %>%
  mutate(
    term_clean = factor(
      term_clean,
      levels = c(main_order, lulc_order)
    )
  )


# Mean and 95% CI

summary_df <- plot_df %>%
  group_by(term_clean) %>%
  summarise(
    n_birds = n(),
    mean_beta = mean(beta, na.rm = TRUE),
    sd_beta = sd(beta, na.rm = TRUE),
    se_beta = sd_beta / sqrt(n_birds),
    ci_low = mean_beta - qt(0.975, df = n_birds - 1) * se_beta,
    ci_high = mean_beta + qt(0.975, df = n_birds - 1) * se_beta,
    .groups = "drop"
  )


#### Violin plot ####

p <- ggplot(plot_df, aes(x = beta, y = term_clean)) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.5
  ) +
  
  geom_violin(
    fill = "white",
    colour = "black",
    linewidth = 0.7,
    trim = FALSE,
    scale = "width"
  ) +
  
  geom_errorbarh(
    data = summary_df,
    aes(
      y = term_clean,
      xmin = ci_low,
      xmax = ci_high
    ),
    inherit.aes = FALSE,
    height = 0.18,
    colour = "black",
    linewidth = 0.8
  ) +
  
  geom_point(
    data = summary_df,
    aes(
      x = mean_beta,
      y = term_clean
    ),
    inherit.aes = FALSE,
    shape = 23,
    size = 3,
    fill = "white",
    colour = "black",
    stroke = 1
  ) +
  
  geom_jitter(
    aes(colour = bird_id),
    height = 0.08,
    width = 0,
    size = 2.2,
    alpha = 0.9
  ) +
  
  labs(
    x = "Individual RSF coefficient",
    y = "Predictor",
    colour = "Bird ID"
  ) +
  
  theme_bw(base_size = 12) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text = element_text(colour = "black"),
    axis.title = element_text(colour = "black"),
    legend.position = "right",
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 8)
  )

p


# Export
ggsave(
  file.path(output_dir, "Figure2.png"),
  p,
  width = 180,
  height = 135,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(output_dir, "Figure2.svg"),
  p,
  width = 180,
  height = 135,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(output_dir, "Figure2.pdf"),
  p,
  width = 180,
  height = 135,
  units = "mm",
  dpi = 600
)

