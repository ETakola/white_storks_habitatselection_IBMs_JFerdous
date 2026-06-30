###Jannatul
###Modified: 16/05/2026
library(tidyverse)
library(patchwork)

base_dir <- "[path]/validation_RSF"

lobo <- read_csv(
  file.path(base_dir, "LOBO_validation_summary.csv"),
  show_col_types = FALSE
)

kfold <- read_csv(
  file.path(base_dir, "blocked_kfold_by_bird_overall_summary.csv"),
  show_col_types = FALSE
)

validation_df <- tibble(
  Validation = c("LOBO", "Blocked k-fold"),
  
  Top1_mean = c(lobo$mean_top1_acc,
                kfold$mean_top1_acc),
  
  Top1_sd = c(lobo$sd_top1_acc,
              kfold$sd_top1_acc),
  
  Top3_mean = c(lobo$mean_top3_acc,
                kfold$mean_top3_acc),
  
  Top3_sd = c(lobo$sd_top3_acc,
              kfold$sd_top3_acc),
  
  Rank_mean = c(lobo$mean_rank,
                kfold$mean_mean_rank),
  
  Rank_sd = c(as.numeric(lobo$sd_rank),
              kfold$sd_mean_rank)
)

accuracy_long <- validation_df %>%
  select(Validation, Top1_mean, Top1_sd, Top3_mean, Top3_sd) %>%
  pivot_longer(
    cols = -Validation,
    names_to = c("Metric", ".value"),
    names_pattern = "(Top[13])_(mean|sd)"
  ) %>%
  mutate(
    Metric = recode(Metric,
                    Top1 = "Top-1 accuracy",
                    Top3 = "Top-3 accuracy")
  )

p1 <- ggplot(accuracy_long,
             aes(x = Validation,
                 y = mean,
                 fill = Metric)) +
  geom_col(position = position_dodge(width = 0.7),
           width = 0.6,
           color = "black",
           linewidth = 0.2) +
  geom_errorbar(aes(ymin = pmax(mean - sd, 0),
                    ymax = mean + sd),
                position = position_dodge(width = 0.7),
                width = 0.18,
                linewidth = 0.4) +
  geom_hline(yintercept = 0.10,
             linetype = "dashed",
             linewidth = 0.5,
             color = "red") +
  scale_y_continuous(limits = c(0, 0.6),
                     expand = expansion(mult = c(0, 0.05))) +
  labs(
    x = NULL,
    y = "Ranking accuracy",
    fill = NULL
  ) +
  theme_classic(base_size = 12)+
  scale_fill_manual(values = c(
    "Top-1 accuracy" = "#8C510A",
    "Top-3 accuracy" = "#01665E"))+ theme(legend.position = "top")+
  theme(
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 12),
    legend.text = element_text(size = 10),
    legend.position = "top"
  )

p1


ggsave(
  file.path(base_dir, "Figure S1.png"),
  p1,
  width = 140,
  height = 120,
  units = "mm",
  dpi = 600
)
ggsave(
  file.path(base_dir, "Figure S1.svg"),
  plot = final_plot,
  width = 140,
  height = 120,
  units = "mm",
  bg = "white",
  device = svglite::svglite,
  limitsize = FALSE
)

ggsave(
  file.path(base_dir, "Figure S1.pdf"),
  plot = p1,
  width = 140,
  height = 120,
  units = "mm",
  device = cairo_pdf,
  bg = "white",
  limitsize = FALSE
)
