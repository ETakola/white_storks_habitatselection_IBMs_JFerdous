###Jannatul
###Modified: 06/04/2026
library(terra)
library(geodata)
####DIRECTORY CREATION####
dir.create("[path]/temp_terra", recursive = TRUE, showWarnings = FALSE)
terraOptions(memfrac = 0.8, tempdir = "E:/stork_publication/temp_terra", progress = 1)

years <- c(1990, 1995, 2000, 2001, 2012, 2018, 2022)

in_dir <- "[path]/"
in_files <- file.path(in_dir, paste0("LC_", years, "_250m.tif"))
stopifnot(all(file.exists(in_files)))

res_ctx <- 250
window_radius_m <- 1200
w <- 2 * ceiling(window_radius_m / res_ctx) + 1
window <- matrix(1, nrow = w, ncol = w)

cell_m <- res_ctx
window_area_m2 <- (w * cell_m) * (w * cell_m)
window_area_ha <- window_area_m2 / 10000

cat("Context resolution:", res_ctx, "m\n")
cat("Window size:", w, "x", w, "cells\n")
cat("Window area:", window_area_ha, "ha\n")

#### OUTPUT FOLDERS ####
out_dir_ed_shei <- file.path(
  in_dir,
  paste0("ED_SHEI_", res_ctx, "m_epsg3035_r", window_radius_m, "m")
)
dir.create(out_dir_ed_shei, recursive = TRUE, showWarnings = FALSE)

out_dir_pd_shdi <- file.path(
  in_dir,
  paste0("PD_SHDI_", res_ctx, "m_epsg3035_r", window_radius_m, "m")
)
dir.create(out_dir_pd_shdi, recursive = TRUE, showWarnings = FALSE)


#### BOUNDARY ####
de <- geodata::gadm(country = "DEU", level = 0, path = tempdir())
ref <- rast(in_files[years == 2018])
de_r <- project(de, crs(ref))

#### HELPERS ####
crop_mask_to_germany <- function(r, de_proj) {
  r <- crop(r, ext(de_proj))
  r <- mask(r, de_proj)
  r
}

prep_lulc_ctx <- function(file, de_proj, template_rast) {
  cat("\nReading:", basename(file), "\n")
  r <- rast(file)
  r <- crop_mask_to_germany(r, de_proj)
  
  same_geom <- compareGeom(r, template_rast, stopOnError = FALSE)
  if (!isTRUE(same_geom)) {
    cat("  Resampling to template grid...\n")
    r <- resample(r, template_rast, method = "near")
  }
  
  as.factor(r)
}


#### TEMPLATE DEFINING ####

template_ctx <- crop_mask_to_germany(ref, de_r)


#### METRIC FUNCTIONS ####
# ---------- ED ----------
edge_length_raster <- function(lulc_ctx, cell_m) {
  east  <- shift(lulc_ctx, dx = 1, dy = 0)
  south <- shift(lulc_ctx, dx = 0, dy = -1)
  
  h_edge <- (lulc_ctx != east)  & !is.na(lulc_ctx) & !is.na(east)
  v_edge <- (lulc_ctx != south) & !is.na(lulc_ctx) & !is.na(south)
  
  ((h_edge * 1) + (v_edge * 1)) * cell_m
}

compute_ed <- function(lulc_ctx, window, window_area_ha, cell_m) {
  cat("  Computing ED...\n")
  elen <- edge_length_raster(lulc_ctx, cell_m = cell_m)
  
  edge_sum <- focal(
    elen,
    w = window,
    fun = "sum",
    na.rm = TRUE,
    fillvalue = NA
  )
  
  ed <- edge_sum / window_area_ha
  names(ed) <- "ED"
  ed
}

# ---------- SHDI + SHEI ----------
compute_shdi_shei_fast <- function(lulc_ctx, window) {
  cat("  Computing SHDI and SHEI...\n")
  
  classes <- unique(values(lulc_ctx))
  classes <- classes[!is.na(classes)]
  classes <- sort(classes)
  
  if (length(classes) == 0) {
    stop("No valid classes found in raster.")
  }
  
  valid <- !is.na(lulc_ctx)
  valid_n <- focal(
    valid,
    w = window,
    fun = "sum",
    na.rm = TRUE,
    fillvalue = NA
  )
  
  shdi <- rast(lulc_ctx)
  values(shdi) <- 0
  names(shdi) <- "SHDI"
  
  s_count <- rast(lulc_ctx)
  values(s_count) <- 0
  
  for (cl in classes) {
    cat("    Class:", cl, "\n")
    
    r_bin <- lulc_ctx == cl
    r_bin[is.na(lulc_ctx)] <- NA
    
    cnt <- focal(
      r_bin,
      w = window,
      fun = "sum",
      na.rm = TRUE,
      fillvalue = NA
    )
    
    p <- cnt / valid_n
    present <- ifel(p > 0, 1, 0)
    term <- ifel(p > 0, -(p * log(p)), 0)
    
    s_count <- s_count + present
    shdi <- shdi + term
    
    rm(r_bin, cnt, p, present, term)
    gc()
  }
  
  shdi <- ifel(valid_n > 0, shdi, NA)
  shei <- ifel(valid_n > 0, ifel(s_count > 1, shdi / log(s_count), 0), NA)
  
  names(shdi) <- "SHDI"
  names(shei) <- "SHEI"
  
  list(shdi = shdi, shei = shei)
}

# ---------- PD ----------
fun_pd <- function(v) {
  v <- matrix(v, nrow = w, ncol = w)
  
  if (all(is.na(v))) return(NA_real_)
  
  rr <- rast(v)
  
  classes <- unique(as.vector(v))
  classes <- classes[!is.na(classes)]
  if (length(classes) == 0) return(NA_real_)
  
  n_patches <- 0
  
  for (cl in classes) {
    rcl <- rr == cl
    p <- patches(rcl, directions = 8)
    ids <- unique(values(p))
    ids <- ids[!is.na(ids) & ids > 0]
    n_patches <- n_patches + length(ids)
  }
  
  pd <- n_patches / window_area_ha * 100
  as.numeric(pd)
}

compute_pd <- function(lulc_ctx, window) {
  cat("  Computing PD...\n")
  pd <- focal(
    lulc_ctx,
    w = window,
    fun = fun_pd,
    na.policy = "omit",
    fillvalue = NA
  )
  names(pd) <- "PD"
  pd
}


#### GROUP 1: ED + SHEI ####

for (i in seq_along(years)) {
  y <- years[i]
  
  cat("\n====================================\n")
  cat("YEAR:", y, "- GROUP 1: ED + SHEI\n")
  cat("====================================\n")
  
  lulc_ctx <- prep_lulc_ctx(in_files[i], de_r, template_ctx)
  
  ed <- compute_ed(
    lulc_ctx = lulc_ctx,
    window = window,
    window_area_ha = window_area_ha,
    cell_m = cell_m
  )
  
  div <- compute_shdi_shei_fast(
    lulc_ctx = lulc_ctx,
    window = window
  )
  
  shei <- div$shei
  
  f_ed   <- file.path(out_dir_ed_shei, paste0("ED_", y, "_", res_ctx, "m.tif"))
  f_shei <- file.path(out_dir_ed_shei, paste0("SHEI_", y, "_", res_ctx, "m.tif"))
  
  writeRaster(ed, f_ed, overwrite = TRUE)
  writeRaster(shei, f_shei, overwrite = TRUE)
  
  cat("Wrote:\n", f_ed, "\n", f_shei, "\n")
  
  rm(lulc_ctx, ed, div, shei)
  gc()
}

#### GROUP 2: PD + SHDI ####

for (i in seq_along(years)) {
  y <- years[i]
  
  cat("\n====================================\n")
  cat("YEAR:", y, "- GROUP 2: PD + SHDI\n")
  cat("====================================\n")
  
  lulc_ctx <- prep_lulc_ctx(in_files[i], de_r, template_ctx)
  
  pd <- compute_pd(
    lulc_ctx = lulc_ctx,
    window = window
  )
  
  div <- compute_shdi_shei_fast(
    lulc_ctx = lulc_ctx,
    window = window
  )
  
  shdi <- div$shdi
  
  f_pd   <- file.path(out_dir_pd_shdi, paste0("PD_", y, "_", res_ctx, "m.tif"))
  f_shdi <- file.path(out_dir_pd_shdi, paste0("SHDI_", y, "_", res_ctx, "m.tif"))
  
  writeRaster(pd, f_pd, overwrite = TRUE)
  writeRaster(shdi, f_shdi, overwrite = TRUE)
  
  cat("Wrote:\n", f_pd, "\n", f_shdi, "\n")
  
  rm(lulc_ctx, pd, div, shdi)
  gc()
}
