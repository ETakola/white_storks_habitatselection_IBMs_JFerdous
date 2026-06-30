####Supplementary section: Figures S3, S4, S5 ####
###Jannatul
###Modified: 06/06/2026
library(tidyverse)
library(patchwork)
library(dplyr)

#####Plot S3: MIN_SHIFT_DISTANCE_M#####
plot_df <- readRDS("[path]/MIN_SHIFT_DISTANCE_M_year_seed_summary.rds")
parameter_name <- "MIN_SHIFT_DISTANCE_M"
out_dir <- "[path]/Figures"
dir.create(out_dir, showWarnings = FALSE)

plot_df <- plot_df %>%
  mutate(
    parameter_value = factor(
      parameter_value,
      levels = c(1000, 3000, 10000)
    ),
    
    replicate = factor(seed),
    
    year = factor(year)
  )

theme_sens <- theme_classic(base_size = 12) +
  theme(
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 12),
    legend.position = "right",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold")
  )
#Plot function
make_plot <- function(df, yvar, ylabel) {
  
  ggplot(
    df,
    aes(
      x = parameter_value,
      y = .data[[yvar]],
      color = year,
      shape = replicate
    )
  ) +
    
    geom_jitter(
      width = 0.10,
      size = 2.5,
      alpha = 0.85
    ) +
    
    stat_summary(
      aes(group = parameter_value),
      fun = mean,
      geom = "point",
      color = "black",
      size = 3.4
    ) +
    
    stat_summary(
      aes(group = parameter_value),
      fun.data = mean_se,
      geom = "errorbar",
      color = "black",
      width = 0.12,
      linewidth = 0.6
    ) +
    
    labs(
      x = parameter_name,
      y = ylabel,
      color = "Year",
      shape = "Replicate"
    )+
    
    theme_sens
}

#Create panels
p1 <- make_plot(
  plot_df,
  "successful_breeders",
  "Successful breeders"
)

p2 <- make_plot(
  plot_df,
  "dead",
  "Dead birds"
)

p3 <- make_plot(
  plot_df,
  "shifter",
  "Shifters"
) +
  scale_y_continuous(
    breaks = seq(
      0,
      max(plot_df$shifter, na.rm = TRUE),
      by = 1
    )
  )

p4 <- make_plot(
  plot_df,
  "mean_cumulative_energy",
  "Mean cumulative energy"
)
#Combine plots
combined_plot <- ((p1 | p2) / (p3 | p4)) +
  plot_layout(guides = "collect") +
  plot_annotation(tag_levels = "a") +
  theme(
    legend.position = "right",
    legend.box = "vertical"
  )

combined_plot

ggsave(
  file.path(out_dir, "Figure S3.png"),
  combined_plot,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(out_dir, "Figure S3.pdf"),
  combined_plot,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(out_dir, "Figure S3.svg"),
  combined_plot,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)

combined_plot


#####Plot S4: MIN_BETTER_SITE_SCORE#####
plot_df <- readRDS(
  "E:/IBM_stork_resources/Sensitivity_validation_final/MIN_BETTER_SITE_SCORE_year_seed_summary.rds"
)

parameter_name <- "MIN_BETTER_SITE_SCORE"

plot_df1 <- plot_df %>%
  mutate(
    parameter_value = factor(
      parameter_value,
      levels = c(0.05, 0.1, 0.2)
    ),
    
    replicate = factor(seed),
    
    year = factor(year)
  )

theme_sens <- theme_classic(base_size = 12) +
  theme(
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 12),
    legend.position = "right",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold")
  )


# PLOT FUNCTION

make_plot <- function(df, yvar, ylabel) {
  
  ggplot(
    df,
    aes(
      x = parameter_value,
      y = .data[[yvar]],
      color = year,
      shape = replicate
    )
  ) +
    
    geom_jitter(
      width = 0.10,
      size = 2.5,
      alpha = 0.85
    ) +
    
    stat_summary(
      aes(group = parameter_value),
      fun = mean,
      geom = "point",
      color = "black",
      size = 3.4
    ) +
    
    stat_summary(
      aes(group = parameter_value),
      fun.data = mean_se,
      geom = "errorbar",
      color = "black",
      width = 0.12,
      linewidth = 0.6
    ) +
    
    labs(
      x = parameter_name,
      y = ylabel,
      color = "Year",
      shape = "Replicate"
    )+
    
    theme_sens
}


# CREATE PANELS

p1 <- make_plot(
  plot_df1,
  "successful_breeders",
  "Successful breeders"
)

p2 <- make_plot(
  plot_df1,
  "dead",
  "Dead birds"
)

p3 <- make_plot(
  plot_df1,
  "shifter",
  "Shifters"
) +
  scale_y_continuous(
    breaks = seq(
      0,
      max(plot_df$shifter, na.rm = TRUE),
      by = 1
    )
  )

p4 <- make_plot(
  plot_df1,
  "mean_cumulative_energy",
  "Mean cumulative energy"
)

# COMBINE
combined_plot <- ((p1 | p2) / (p3 | p4)) +
  plot_layout(guides = "collect") +
  plot_annotation(tag_levels = "a") +
  theme(
    legend.position = "right",
    legend.box = "vertical"
  )

combined_plot

ggsave(
  file.path(out_dir, "Figure S4.png"),
  combined_plot,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(out_dir, "Figure S4.pdf"),
  combined_plot,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(out_dir, "Figure S4.svg"),
  combined_plot,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)

combined_plot

#####Plot S5: Breed_Min_excess#####
plot_df <- readRDS(
  "E:/IBM_stork_resources/Sensitivity_validation_final/BREED_MIN_EXCESS_year_seed_summary.rds"
)
parameter_name <- "BREED_MIN_EXCESS"


plot_df1 <- plot_df %>%
  mutate(
    parameter_value = factor(
      parameter_value,
      levels = c(0.1, 0.3, 0.5)
    ),
    
    replicate = factor(seed),
    
    year = factor(year)
  )

theme_sens <- theme_classic(base_size = 12) +
  theme(
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 12),
    legend.position = "right",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold")
  )


# PLOT FUNCTION
make_plot <- function(df, yvar, ylabel) {
  
  ggplot(
    df,
    aes(
      x = parameter_value,
      y = .data[[yvar]],
      color = year,
      shape = replicate
    )
  ) +
    
    geom_jitter(
      width = 0.10,
      size = 2.5,
      alpha = 0.85
    ) +
    
    stat_summary(
      aes(group = parameter_value),
      fun = mean,
      geom = "point",
      color = "black",
      size = 3.4
    ) +
    
    stat_summary(
      aes(group = parameter_value),
      fun.data = mean_se,
      geom = "errorbar",
      color = "black",
      width = 0.12,
      linewidth = 0.6
    ) +
    
    labs(
      x = parameter_name,
      y = ylabel,
      color = "Year",
      shape = "Replicate"
    )+
    
    theme_sens
}

# CREATE PANELS
p1 <- make_plot(
  plot_df1,
  "successful_breeders",
  "Successful breeders"
)

p2 <- make_plot(
  plot_df1,
  "dead",
  "Dead birds"
)

p3 <- make_plot(
  plot_df1,
  "shifter",
  "Shifters"
) +
  scale_y_continuous(
    breaks = seq(
      0,
      max(plot_df$shifter, na.rm = TRUE),
      by = 1
    )
  )

p4 <- make_plot(
  plot_df1,
  "mean_cumulative_energy",
  "Mean cumulative energy"
)

#COMBINE
combined_plot <- ((p1 | p2) / (p3 | p4)) +
  plot_layout(guides = "collect") +
  plot_annotation(tag_levels = "a") +
  theme(
    legend.position = "right",
    legend.box = "vertical"
  )

combined_plot

ggsave(
  file.path(out_dir, "Figure S5.png"),
  combined_plot,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(out_dir, "Figure S5.pdf"),
  combined_plot,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(out_dir, "Figure S5.svg"),
  combined_plot,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)

combined_plot
