###Jannatul
###Modified: 31/03/2026
####Used-avail points creation####
setwd("E:/IBM_stork_resources/Steps_output")
library(dplyr)
library(lubridate)
library(geosphere)

df_used <- read.csv("[path]/MPIAB_inside_data.csv")
# timestamp formatting to POSIXct
df_used$timestamp <- ymd_hms(df_used$timestamp)
df_used$year  <- year(df_used$timestamp)
df_used$month <- month(df_used$timestamp)
df_used$bird_id<- df_used$individual.local.identifier
df_used$lat<-df_used$location.lat
df_used$lon<-df_used$location.long
####1. CREATE USED ROWS ####
df_used_out <- df_used %>%
  transmute(
    bird_id,
    lat,
    lon,
    Used      = 1,
    year,
    timestamp,
    month
  )

####2.GENERATE 10 AVAILABLE POINTS PER USED POINT####
set.seed(123)

available_list <- lapply(1:nrow(df_used), function(i) {
  
  u <- df_used[i, ]
  
  bearings  <- runif(10, 0, 360)
  distances <- runif(10, 1e3, 50e3)
  
  pts <- destPoint(
    p = c(u$lon, u$lat),
    b = bearings,
    d = distances
  )
  
  data.frame(
    bird_id   = u$bird_id,
    lat       = pts[,2],
    lon       = pts[,1],
    Used      = 0,
    year      = u$year,
    timestamp = u$timestamp,
    month     = u$month
  )
})

df_available <- do.call(rbind, available_list)

####3. COMBINE USED + AVAILABLE ####
df_rsf <- bind_rows(df_used_out, df_available)

write.csv(df_rsf, "used_avaiable_datapoints.csv", row.names = FALSE)
saveRDS(df_rsf, "used_avaiable_datapoints.rds")


