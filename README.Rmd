## Details

- Maps of NDVI, NDWI and land use are extracted from Google Earth Engine.
- RSFs are fitted with a mixed-effects RSF with glmmTMB R package.
- IBM was done in Python.

<!-- -->

## Bird tracks

```{r echo=FALSE, results='asis'}
geo <- paste(readLines("bird_tracks_lines.geojson", warn = FALSE), collapse = "\n")
cat("\n```geojson\n", geo, "\n```\n", sep = "")
