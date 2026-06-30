###Jannatul
###Modified: 11/06/2026
library(tidyverse)
library(ggnewscale)

energy_df <- readRDS("[path]/merged_summary_all_years.rds")

# prepare plotting data
plot_energy <- energy_df %>%
  mutate(
    year = factor(year, levels = sort(unique(year))),
    bird_id = factor(bird_id, levels = rev(sort(unique(bird_id)))),
    deceased = alive == FALSE | final_state == "dead"
  )

# plot
p_energy <- ggplot() +
  geom_tile(
    data = plot_energy %>% filter(!deceased),
    aes(
      x = year,
      y = bird_id,
      fill = energy_cumulative
    ),
    colour = "white",
    linewidth = 0.35
  ) +
  scale_fill_gradient(
    low = "#d8e6e6",
    high = "#d95f02",
    name = "Energy score",
    limits = c(0, max(plot_energy$energy_cumulative, na.rm = TRUE)),
    na.value = "white"
  ) +
  ggnewscale::new_scale_fill() +
  geom_tile(
    data = plot_energy %>% filter(deceased),
    aes(
      x = year,
      y = bird_id,
      fill = "Deceased"
    ),
    colour = "white",
    linewidth = 0.35
  ) +
  scale_fill_manual(
    name = NULL,
    values = c("Deceased" = "black")
  ) +
  labs(
    #title = "Energy budget\nper individual per year",
    x = "Year",
    y = "Bird ID"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(size = 18, hjust = 0.5, lineheight = 0.95),
    axis.text.x = element_text(angle = 45, hjust = 1, colour = "black"),
    axis.text.y = element_text(colour = "black"),
    axis.title = element_text(colour = "black"),
    panel.grid = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10)
  )

p_energy

fig_dir <- "E:/IBM_stork_resources/figures"

ggsave(
  file.path(fig_dir, "Figure 6.svg"),
  p_energy,
  width = 100,
  height = 190,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(fig_dir, "Figure 6.pdf"),
  p_energy,
  width = 100,
  height = 190,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(fig_dir, "Figure 6.png"),
  p_energy,
  width = 100,
  height = 190,
  units = "mm",
  dpi = 600
)
