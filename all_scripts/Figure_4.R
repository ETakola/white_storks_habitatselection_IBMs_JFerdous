###Jannatul
###Modified: 10/06/2026
library(tidyverse)
library(viridis)
output_dir <- "[path]/figures"
df <- readRDS("[path]/merged_simulated_bird_steps.rds")

grid_size <- 0.03
activity_grid <- df %>%
  mutate(
    final_state = str_replace_all(final_state, "_", " ") %>% str_to_title(),
    lon_grid = round(lon / grid_size) * grid_size,
    lat_grid = round(lat / grid_size) * grid_size
  ) %>%
  group_by(final_state, lon_grid, lat_grid) %>%
  summarise(
    weighted_activity = n(),
    .groups = "drop"
  ) %>%
  filter(weighted_activity > 0)



plot3 <- ggplot(activity_grid, aes(x = lon_grid, y = lat_grid, colour = weighted_activity)) +
  geom_point(shape = 15, size = 0.1, alpha = 1) +
  facet_wrap(~ final_state, ncol = 3) +
  scale_colour_viridis_c(
    option = "D",
    trans = "log10",
    name = "Weighted activity\n(sum of steps per grid cell)",
    breaks = c(10, 100, 1000, 10000, 100000),
    labels = c(
      expression(10^1),
      expression(10^2),
      expression(10^3),
      expression(10^4),
      expression(10^5)
    )
  ) +
  scale_x_continuous(limits = c(10, 15), breaks = 10:15) +
  scale_y_continuous(limits = c(50.4, 54.1), breaks = 51:54) +
  labs(
    #title = "Activity map by final state",
    x = "Longitude",
    y = "Latitude"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold"),
    panel.grid = element_line(colour = "grey90", linewidth = 0.3),
    panel.border = element_blank(),
    legend.position = "right"
  )
plot3
# Export
ggsave(
  file.path(output_dir, "Figure 4.png"),
  plot3,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(output_dir, "Figure 4.svg"),
  plot3,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)

ggsave(
  file.path(output_dir, "Figure 4.pdf"),
  plot3,
  width = 190,
  height = 120,
  units = "mm",
  dpi = 600
)
