#!/usr/bin/env Rscript
# Build-time conversion of ExPanDaR's R list configs to JSON.
#
# Data frames are converted separately in Python via pyreadr (see tools/build_datasets.py);
# this script only handles the two `ExPanD_config_*` objects, which are R *lists* and
# therefore cannot be read by pyreadr.
#
# Usage: Rscript tools/convert_rdata.R <expandar_data_dir> <out_dir>

suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[[1]] else "ExPanDaR/data"
out_dir <- if (length(args) >= 2) args[[2]] else "src/expdpy/data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

convert_config <- function(rdata, obj, out) {
  e <- new.env()
  load(file.path(data_dir, rdata), envir = e)
  cfg <- e[[obj]]
  writeLines(
    jsonlite::toJSON(cfg, auto_unbox = TRUE, null = "null", na = "null", pretty = TRUE),
    file.path(out_dir, out)
  )
  cat("wrote", out, "\n")
}

convert_config("ExPanD_config_russell_3000.RData", "ExPanD_config_russell_3000",
               "ExPanD_config_russell_3000.json")
convert_config("ExPanD_config_worldbank.RData", "ExPanD_config_worldbank",
               "ExPanD_config_worldbank.json")
