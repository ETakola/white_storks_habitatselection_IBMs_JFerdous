####Figure 3####
###Jannatul
###Modified: 26/05/2026

library(sf)
library(dplyr)
library(ggplot2)
library(lubridate)
library(cowplot)
library(scales)
library(rnaturalearth)
library(rnaturalearthdata)
library(geodata)
library(ggspatial)
library(grid)

####SET GLOBAL PLOT PARAMETERS####
theme_set(theme_minimal(base_size = 7))

plt_base_family <- "Arial"

theme_update(
  text = element_text(family = plt_base_family, colour = "black"),
  plot.title = element_text(size = 7, face = "bold", hjust = 0),
  axis.text = element_text(size = 6, colour = "black"),
  axis.title = element_text(size = 6, colour = "black"),
  panel.grid = element_blank(),
  legend.text = element_text(size = 6),
  legend.title = element_text(size = 6, face = "bold")
)


#### LOAD DATA ####
bird_df <- readRDS("[path]/merged_simulated_bird_steps.rds")

col_bird_id   <- "bird_id"
col_timestamp <- "timestamp"
col_year      <- "year"
col_lon       <- "lon"
col_lat       <- "lat"
col_state     <- "final_state"

target_crs <- 3035
####DATA FORMATTING####
bird_df <- bird_df %>%
  mutate(
    bird_id     = as.character(.data[[col_bird_id]]),
    final_state = as.character(.data[[col_state]]),
    year        = as.integer(.data[[col_year]])
  )

if (!inherits(bird_df[[col_timestamp]], c("POSIXct", "POSIXt"))) {
  bird_df[[col_timestamp]] <- suppressWarnings(ymd_hms(bird_df[[col_timestamp]], tz = "UTC"))
  if (all(is.na(bird_df[[col_timestamp]]))) {
    bird_df[[col_timestamp]] <- suppressWarnings(parse_date_time(
      bird_df[[col_timestamp]],
      orders = c("ymd HMS", "ymd HM", "dmy HMS", "dmy HM", "mdy HMS", "mdy HM"),
      tz = "UTC"
    ))
  }
}

state_levels <- c("successful_breeder", "failed_breeder", "non_breeder", "shifter", "dead")

bird_df <- bird_df %>%
  filter(
    !is.na(.data[[col_lon]]),
    !is.na(.data[[col_lat]]),
    !is.na(.data[[col_year]]),
    !is.na(.data[[col_bird_id]]),
    !is.na(.data[[col_state]]),
    !is.na(.data[[col_timestamp]])
  ) %>%
  filter(final_state %in% state_levels) %>%
  arrange(year, bird_id, .data[[col_timestamp]])


#### COLORS ####

state_cols <- c(
  "dead"               = "black",
  "failed_breeder"     = "#D55E00",
  "non_breeder"        = "#009E73",
  "shifter"            = "#0072B2",
  "successful_breeder" = "#E69F00"
)

state_labels <- c(
  "successful_breeder" = "Successful breeder",
  "failed_breeder"     = "Failed breeder",
  "non_breeder"        = "Non-breeder",
  "shifter"            = "Shifter",
  "dead"               = "Dead"
)


#### GERMAN ADMINISTRATIVE DIVISIONS ####

germany <- ne_countries(scale = "medium", returnclass = "sf") %>%
  filter(admin == "Germany") %>%
  st_transform(target_crs)

germany_states <- geodata::gadm(
  country = "DEU",
  level = 1,
  path = tempdir()
) %>%
  st_as_sf() %>%
  st_transform(target_crs)

bbox <- st_bbox(germany)

bird_pts <- st_as_sf(
  bird_df,
  coords = c(col_lon, col_lat),
  crs = 4326,
  remove = FALSE
) %>%
  st_transform(target_crs)


#### TRAJECTORIES SET ####

bird_lines <- bird_pts %>%
  group_by(year, bird_id) %>%
  summarise(
    final_state = dplyr::last(final_state),
    n_points = dplyr::n(),
    do_union = FALSE,
    .groups = "drop"
  ) %>%
  filter(n_points > 1) %>%
  st_cast("LINESTRING")

years <- sort(unique(bird_lines$year))

if (length(years) > 5) {
  warning("More than 5 years detected. Only the first 5 sorted years will be plotted.")
  years <- years[1:5]
}

#### BIRD COLORS ####
all_birds <- sort(unique(as.character(bird_lines$bird_id)))
n_birds <- length(all_birds)

manual_bird_palette <- c(
  "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
  "#a6761d", "#666666", "#1f78b4", "#b2df8a", "#fb9a99", "#fdbf6f",
  "#cab2d6", "#6a3d9a", "#ff7f00", "#b15928", "#17becf", "#bcbd22",
  "#8c564b", "#9467bd", "#2ca02c", "#ff9896", "#98df8a", "#aec7e8",
  "#c5b0d5", "#c49c94", "#9edae5", "#dbdb8d", "#393b79", "#637939",
  "#8c6d31", "#843c39", "#7b4173"
)

if (n_birds > length(manual_bird_palette)) {
  stop("More birds than available colors in manual_bird_palette.")
}

bird_cols <- setNames(manual_bird_palette[seq_len(n_birds)], all_birds)


#### STATE SUMMARY ####

state_summary <- bird_df %>%
  distinct(year, bird_id, final_state) %>%
  group_by(year, bird_id) %>%
  summarise(final_state = dplyr::last(final_state), .groups = "drop") %>%
  count(year, final_state, name = "n")

state_summary <- expand.grid(
  year = years,
  final_state = state_levels,
  stringsAsFactors = FALSE
) %>%
  left_join(state_summary, by = c("year", "final_state")) %>%
  mutate(
    n = ifelse(is.na(n), 0L, n),
    final_state = factor(final_state, levels = state_levels)
  )


#### INSET BARPLOT ####
make_inset <- function(y) {
  df <- state_summary %>%
    filter(year == y) %>%
    mutate(
      label = factor(
        state_labels[as.character(final_state)],
        levels = state_labels[state_levels]
      )
    )
  
  ymax <- max(df$n, na.rm = TRUE)
  if (!is.finite(ymax) || ymax == 0) ymax <- 1
  
  ggplot(df, aes(x = label, y = n, fill = final_state)) +
    geom_col(width = 0.58, color = "black", linewidth = 0.18) +
    scale_fill_manual(values = state_cols, drop = FALSE) +
    scale_y_continuous(
      expand = expansion(mult = c(0, 0.05)),
      breaks = pretty(c(0, ymax), n = 3)
    ) +
    coord_cartesian(clip = "off") +
    labs(x = NULL, y = NULL) +
    theme_minimal(base_size = 5.6) +
    theme(
      legend.position = "none",
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "grey88", linewidth = 0.15),
      axis.text.x = element_blank(),
      axis.text.y = element_text(size = 4.8, colour = "black", margin = margin(r = 1)),
      axis.ticks = element_blank(),
      axis.title = element_blank(),
      panel.background = element_rect(fill = "white", color = "black", linewidth = 0.25),
      plot.background  = element_rect(fill = "white", color = "black", linewidth = 0.25),
      plot.margin = margin(0, 0, 4, 10)
    )
}


#### MAP PANEL####
make_map_panel <- function(y, add_north_scale = TRUE) {
  lines_y <- bird_lines %>% filter(year == y)
  
  xr <- as.numeric(bbox["xmax"] - bbox["xmin"])
  yr <- as.numeric(bbox["ymax"] - bbox["ymin"])
  
  p <- ggplot() +
    geom_sf(data = germany, fill = "#f4f2ec", color = "#8c8c8c", linewidth = 0.22) +
    geom_sf(data = germany_states, fill = NA, color = "#bdbdbd", linewidth = 0.18) +
    geom_sf(
      data = lines_y,
      aes(color = bird_id),
      linewidth = 0.34,
      alpha = 0.34,
      show.legend = FALSE
    ) +
    scale_color_manual(values = bird_cols, guide = "none") +
    coord_sf(
      xlim = c(bbox["xmin"] - 0.025 * xr, bbox["xmax"] + 0.025 * xr),
      ylim = c(bbox["ymin"] - 0.025 * yr, bbox["ymax"] + 0.025 * yr),
      expand = FALSE
    ) +
    theme_void(base_size = 7) +
    theme(
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      plot.margin = margin(0, 0, 0, 0)
    )
  
  if (add_north_scale) {
    p <- p +
      annotation_north_arrow(
        location = "tr",
        which_north = "true",
        pad_x = unit(0.03, "in"),
        pad_y = unit(0.18, "in"),
        height = unit(0.18, "in"),
        width = unit(0.18, "in"),
        style = north_arrow_orienteering(
          line_width = 0.4,
          text_size = 5
        )
      ) +
      annotation_scale(
        location = "br",
        pad_x = unit(0.03, "in"),
        pad_y = unit(0.03, "in"),
        width_hint = 0.22,
        line_width = 0.35,
        text_cex = 0.45
      )
  }
  
  p
}


# YEAR PANEL WITH TRUE INSET WITH EXPLICIT BORDER
make_year_panel <- function(y, panel_label = NULL) {
  p_map <- make_map_panel(y)
  p_bar <- make_inset(y)
  
  ggdraw() +
    draw_plot(p_map, x = 0, y = 0, width = 1, height = 1) +
    draw_plot(
      p_bar,
      x = 0.02, y = 0.02,
      width = 0.30, height = 0.28
    ) +
    draw_label(
      paste0(panel_label, "  ", y),
      x = 0.02, y = 0.98,
      hjust = 0, vjust = 1,
      fontface = "bold",
      size = 7
    ) +
    draw_grob(
      rectGrob(
        gp = gpar(col = "black", fill = NA, lwd = 0.8)
      ),
      x = 0, y = 0, width = 1, height = 1
    )
}


# COMBINED LEGEND PANEL
make_combined_legend_panel <- function(panel_label = NULL) {
#bird ID layout 15-9-9 in legend
  if (length(all_birds) != 33) {
    stop("This custom legend layout expects exactly 33 birds.")
  }
  
  birds_col1 <- all_birds[1:9]
  birds_col2 <- all_birds[10:18]
  birds_col3 <- all_birds[19:33]
  
  make_bird_col_df <- function(ids, col_x) {
    data.frame(
      bird_id = ids,
      x = col_x,
      y = rev(seq_along(ids))
    )
  }
  
  bird_leg_df <- bind_rows(
    make_bird_col_df(birds_col1, 1),
    make_bird_col_df(birds_col2, 2),
    make_bird_col_df(birds_col3, 3)
  )
  
  bird_leg_df <- bird_leg_df %>%
    group_by(x) %>%
    mutate(y_plot = y + (15 - max(y))) %>%
    ungroup()
  
  p_birds <- ggplot(bird_leg_df, aes(x = x, y = y_plot)) +
    geom_segment(
      aes(
        x = x - 0.12, xend = x + 0.12,
        y = y_plot, yend = y_plot,
        color = bird_id
      ),
      linewidth = 0.9,
      lineend = "round"
    ) +
    geom_text(
      aes(x = x + 0.19, label = bird_id),
      hjust = 0,
      size = 1.9,
      family = plt_base_family
    ) +
    scale_color_manual(values = bird_cols, guide = "none") +
    coord_cartesian(
      xlim = c(0.72, 3.85),
      ylim = c(0.2, 16.1),
      clip = "off"
    ) +
    annotate(
      "text",
      x = 0.76, y = 15.75,
      label = "Bird ID",
      hjust = 0,
      size = 2.5,
      fontface = "bold",
      family = plt_base_family
    ) +
    theme_void() +
    theme(
      plot.margin = margin(8, 6, 6, 6),
      plot.background = element_rect(fill = "white", color = NA)
    )
  
  state_leg_df <- data.frame(
    final_state = factor(state_levels, levels = state_levels),
    x = 1,
    y = c(5.0, 3.9, 2.8, 1.7, 0.6),
    short = c("Successful-br", "Failed br", "Non-br", "Shifter", "Dead")
  )
  
  p_state_inset <- ggplot(state_leg_df, aes(x = x, y = y)) +
    geom_tile(
      aes(fill = final_state),
      width = 0.13, height = 0.34,
      color = "black", linewidth = 0.18
    ) +
    geom_text(
      aes(x = 1.13, label = short),
      hjust = 0,
      size = 1.75,
      family = plt_base_family
    ) +
    scale_fill_manual(values = state_cols, guide = "none") +
    coord_cartesian(
      xlim = c(0.88, 1.75),
      ylim = c(0.2, 6.3),
      clip = "off"
    ) +
    annotate(
      "text",
      x = 0.90, y = 5.95,
      label = "Bird status",
      hjust = 0,
      size = 2.0,
      fontface = "bold",
      family = plt_base_family
    ) +
    theme_void() +
    theme(
      plot.background = element_rect(fill = "white", color = "black", linewidth = 0.22),
      panel.background = element_rect(fill = "white", color = "black", linewidth = 0.22),
      plot.margin = margin(2, 2, 2, 2)
    )
  
  ggdraw() +
    draw_plot(p_birds, x = 0, y = 0, width = 1, height = 1) +
    draw_plot(
      p_state_inset,
      x = 0.1, y = 0.03,   # kept exactly as you set
      width = 0.40, height = 0.38
    ) +
    draw_label(
      paste0(panel_label, "  Legend"),
      x = 0.02, y = 0.98,
      hjust = 0, vjust = 1,
      fontface = "bold",
      size = 7
    ) +
    draw_grob(
      rectGrob(
        gp = gpar(col = "black", fill = NA, lwd = 0.8)
      ),
      x = 0, y = 0, width = 1, height = 1
    )
}

#### BUILD PANELS ####
panel_tags <- c("(a)", "(b)", "(c)", "(d)", "(e)", " ")

year_panels <- lapply(seq_along(years), function(i) {
  make_year_panel(years[i], panel_label = panel_tags[i])
})

legend_panel <- make_combined_legend_panel(panel_label = panel_tags[length(years) + 1])

all_panels <- c(year_panels, list(legend_panel))

final_plot <- plot_grid(
  plotlist = all_panels,
  ncol = 3,
  align = "hv",
  rel_widths = c(1, 1, 1),
  rel_heights = c(1, 1)
)


#### EXPORT ####

ggsave(
  filename = "E:/IBM_stork_resources/Figure 3.svg",
  plot = final_plot,
  width = 190,
  height = 160,
  units = "mm",
  bg = "white",
  device = svglite::svglite,
  limitsize = FALSE
)

ggsave(
  filename = "E:/IBM_stork_resources/Figure 3.pdf",
  plot = final_plot,
  width = 190,
  height = 160,
  units = "mm",
  device = cairo_pdf,
  bg = "white",
  limitsize = FALSE
)

ggsave(
  filename = "E:/IBM_stork_resources/Figure 3.png",
  plot = final_plot,
  width = 190,
  height = 160,
  units = "mm",
  dpi = 600,
  bg = "white",
  limitsize = FALSE
)

print(final_plot)

