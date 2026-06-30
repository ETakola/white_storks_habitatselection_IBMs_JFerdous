###Jannatul
###Modified: 11/06/2026
library(tidyverse)
library(viridis)


df <- readRDS("[path]/merged_simulated_bird_steps.rds")

grid_size <- 0.03

state_levels <- c(
  "dead",
  "failed_breeder",
  "non_breeder",
  "shifter",
  "successful_breeder"
)

state_labels <- c(
  "dead" = "Dead",
  "failed_breeder" = "Failed Breeder",
  "non_breeder" = "Non Breeder",
  "shifter" = "Shifter",
  "successful_breeder" = "Successful Breeder"
)

activity_grid_year_state <- df %>%
  filter(
    !is.na(lon),
    !is.na(lat),
    !is.na(year),
    !is.na(final_state)
  ) %>%
  mutate(
    year = factor(year, levels = sort(unique(year))),
    final_state = factor(final_state, levels = state_levels),
    lon_grid = round(lon / grid_size) * grid_size,
    lat_grid = round(lat / grid_size) * grid_size
  ) %>%
  group_by(year, final_state, lon_grid, lat_grid) %>%
  summarise(
    weighted_activity = n(),
    .groups = "drop"
  ) %>%
  filter(weighted_activity > 0)

p_year_state <- ggplot(
  activity_grid_year_state,
  aes(x = lon_grid, y = lat_grid, colour = weighted_activity)
) +
  geom_point(shape = 15, size = 0.3, alpha = 1) +
  facet_grid(
    rows = vars(final_state),
    cols = vars(year),
    labeller = labeller(final_state = state_labels)
  ) +
  scale_colour_viridis_c(
    option = "D",
    trans = "log10",
    name = "Weighted activity\n(sum of steps \nper grid cell)",
    breaks = c(1, 10, 100, 1000, 10000),
    labels = c(
      expression(10^0),
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
    x = "Longitude",
    y = "Latitude"
  ) +
  theme_bw(base_size = 10) +
  theme(
    plot.title = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_text(face = "bold", size = 8, colour = "black"),
    strip.text.y = element_text(face = "bold", size = 8, colour = "black", angle = 90),
    panel.grid.major = element_line(colour = "grey90", linewidth = 0.25),
    panel.grid.minor = element_line(colour = "grey95", linewidth = 0.15),
    panel.border = element_blank(),
    axis.text = element_text(size = 8, colour = "black"),
    axis.title = element_text(size = 8, colour = "black"),
    legend.position = "right",
    legend.title = element_text(size = 7),
    legend.text = element_text(size = 7),
    panel.spacing = unit(0.3, "lines")
  )

p_year_state
fig_dir <- "E:/IBM_stork_resources/figures"

ggsave(
  file.path(fig_dir, "Figure S2.png"),
  p_year_state,
  width = 190,
  height = 160,
  units = "mm",
  dpi = 600,
  bg = "white"
)

ggsave(
  file.path(fig_dir, "Figure S2.pdf"),
  p_year_state,
  width = 190,
  height = 160,
  units = "mm",
  device = cairo_pdf,
  bg = "white"
)

ggsave(
  file.path(fig_dir, "Figure S2.svg"),
  p_year_state,
  width = 190,
  height = 170,
  units = "mm",
  device = svglite::svglite,
  bg = "white"
)

