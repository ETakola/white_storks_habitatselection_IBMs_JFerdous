###Jannatul
###Modified: 10/06/2026
library(tidyverse)
library(viridis)

output_dir <- "[path]/figures"
df <- readRDS("[path]/merged_simulated_bird_steps.rds")

# grid resolution
grid_size <- 0.03

activity_grid_year <- df %>%
  mutate(
    lon_grid = round(lon / grid_size) * grid_size,
    lat_grid = round(lat / grid_size) * grid_size,
    year = as.factor(year)
  ) %>%
  group_by(year, lon_grid, lat_grid) %>%
  summarise(
    weighted_activity = n(),   
    .groups = "drop"
  ) %>%
  filter(weighted_activity > 0)

p_year <- ggplot(
  activity_grid_year,
  aes(x = lon_grid, y = lat_grid, colour = weighted_activity)
) +
  geom_point(shape = 15, size = 0.3, alpha = 1) +
  facet_wrap(~ year, ncol = 3) +
  scale_colour_viridis_c(
    option = "D",
    trans = "log10",
    name = "Weighted activity\n(sum of steps per grid cell)",
    breaks = c(10, 100, 1000, 10000),
    labels = c(
      expression(10^1),
      expression(10^2),
      expression(10^3),
      expression(10^4)
    )
  ) +
  scale_x_continuous(
    limits = c(10, 15),
    breaks = 10:15
  ) +
  scale_y_continuous(
    limits = c(50.4, 54.1),
    breaks = 51:54
  ) +
  labs(
    #title = "Activity map by year",
    x = "Longitude",
    y = "Latitude"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 18, hjust = 0),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 11),
    panel.grid.major = element_line(colour = "grey88", linewidth = 0.35),
    panel.grid.minor = element_line(colour = "grey94", linewidth = 0.25),
    panel.border = element_blank(),
    axis.text = element_text(colour = "black"),
    axis.title = element_text(colour = "black"),
    legend.position = "right",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9)
  )

p_year


ggsave(
  file.path(output_dir, "Figure 5.svg"),
  p_year,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(output_dir, "Figure 5.pdf"),
  p_year,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(output_dir, "Figure 5.png"),
  p_year,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)


