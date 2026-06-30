###Jannatul
###Modified: 31/03/2026
####Landscapemetrics point value extraction####
library(terra)
library(landscapemetrics)
library(dplyr)
####Load Data File####
df <- readRDS("[path]/final_clean_dataset_updated.rds")

target_crs <- "EPSG:3035"
window_radius_m <- 1200   

in_dir  <- "[path]"
out_dir <- "[path]"
cache_dir <- file.path(out_dir, "lulc_cache_epsg3035_250m")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

dir.create("[path]/temp_terra", recursive = TRUE)
terraOptions(memfrac = 0.8, tempdir = "[path]/temp_terra", progress = 1)

#### YEAR MAPPING ####
map_year <- function(y) {
  y <- as.integer(y)
  
  if (y == 1992L) return(1990L)
  if (y %in% c(1994L, 1995L, 1996L, 1997L)) return(1995L)
  if (y %in% c(1998L, 1999L)) return(2000L)
  if (y == 2000L) return(2000L)
  if (y == 2001L) return(2001L)
  if (y == 2012L) return(2012L)
  
  stop("Year not mapped: ", y)
}

df$lulc_year_use <- vapply(df$year, map_year, integer(1))
unique_lulc_years <- sort(unique(df$lulc_year_use))

####POINTS####
pts <- vect(df, geom = c("lon", "lat"), crs = "EPSG:4326")
pts <- project(pts, target_crs)

# CACHE: PROJECT EACH LULC YEAR ONCE TO EPSG:3035
# KEEP NATIVE 250 m RESOLUTION
get_lulc_250m_cached <- function(lulc_year) {
  
  f_cache <- file.path(cache_dir, paste0("LC_", lulc_year, "_EPSG3035_250m.tif"))
  if (file.exists(f_cache)) {
    r <- rast(f_cache)
    return(as.factor(r))
  }
  
  message("Caching LULC year ", lulc_year, " to EPSG:3035 at native 250 m resolution...")
  f_in <- file.path(in_dir, paste0("LC_", lulc_year, "_250m.tif")) #file name pattern
  if (!file.exists(f_in)) stop("Missing input raster: ", f_in)
  
  r <- rast(f_in)
  
  # project only if CRS differs
  if (crs(r) != target_crs) {
    r <- project(r, target_crs, method = "near")
  }
  
  r <- round(r)
  r <- as.factor(r)
  
  writeRaster(r, f_cache, overwrite = TRUE)
  return(as.factor(rast(f_cache)))
}


#### METRICS AROUND ONE POINT ####
compute_point_metrics <- function(lulc_raster, point_geom, radius) {
  
  buf <- buffer(point_geom, width = radius)
  
  r_crop <- crop(lulc_raster, buf)
  r_crop <- mask(r_crop, buf)
  r_crop <- as.factor(r_crop)
  
  if (is.null(r_crop) || ncell(r_crop) == 0) {
    return(c(SHDI = NA_real_, SHEI = NA_real_, ED = NA_real_, PD = NA_real_))
  }
  
  vals <- values(r_crop, na.rm = FALSE)
  if (all(is.na(vals))) {
    return(c(SHDI = NA_real_, SHEI = NA_real_, ED = NA_real_, PD = NA_real_))
  }
  
  shdi <- tryCatch(lsm_l_shdi(r_crop)$value, error = function(e) NA_real_)
  shei <- tryCatch(lsm_l_shei(r_crop)$value, error = function(e) NA_real_)
  ed   <- tryCatch(lsm_l_ed(r_crop)$value,   error = function(e) NA_real_)
  pd   <- tryCatch(lsm_l_pd(r_crop)$value,   error = function(e) NA_real_)
  
  c(SHDI = shdi, SHEI = shei, ED = ed, PD = pd)
}


#### COMPUTE BY LULC YEAR####
results_list <- list()

for (ly in unique_lulc_years) {
  
  cat("\n============================\nLULC year:", ly, "\n============================\n")
  
  r250 <- get_lulc_250m_cached(ly)
  
  idx <- which(df$lulc_year_use == ly)
  pts_sub <- pts[idx]
  
  met <- t(sapply(seq_along(idx), function(i) {
    compute_point_metrics(r250, pts_sub[i], window_radius_m)
  }))
  
  results_list[[as.character(ly)]] <- data.frame(
    row_id = idx,
    met
  )
}

metrics_df <- bind_rows(results_list)


#### MERGE BACK INTO df####
df$SHDI <- NA_real_
df$SHEI <- NA_real_
df$ED   <- NA_real_
df$PD   <- NA_real_

df[metrics_df$row_id, c("SHDI", "SHEI", "ED", "PD")] <-
  metrics_df[, c("SHDI", "SHEI", "ED", "PD")]

# RE-EXTRACT TRUE LULC CLASS
# FROM RAW RASTER ONLY

df$LULC_class <- NA_integer_

for (ly in unique_lulc_years) {
  
  cat("\nExtracting TRUE LULC class for year:", ly, "\n")
  
  f_in <- file.path(in_dir, paste0("LC_", ly, "_250m.tif"))
  if (!file.exists(f_in)) stop("Missing input raster: ", f_in)
  
  r_raw <- rast(f_in)
  
  if (crs(r_raw) != target_crs) {
    r_raw <- project(r_raw, target_crs, method = "near")
  }
  
  r_raw <- round(r_raw)
  
  idx <- which(df$lulc_year_use == ly)
  pts_sub <- pts[idx]
  
  vals <- terra::extract(r_raw, pts_sub)[, 2]
  
  df$LULC_class[idx] <- as.integer(vals)
}

sort(unique(df$LULC_class))

#### SAVE OUTPUTS (rds and csv)####
saveRDS(df, file.path(out_dir, "All_variables_RSF.rds"))
write.csv(df, file.path(out_dir, "All_variables_RSF.csv"), row.names = FALSE)

cat("\nDONE.\nSaved:\n",
    file.path(out_dir, "All_variables_RSF.rds"), "\n",
    file.path(out_dir, "All_variables_RSF.csv"), "\n")

