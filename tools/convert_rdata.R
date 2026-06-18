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

# ExPanDaR's russell_3000 / worldbank config lists are no longer bundled by expdpy, so there
# are currently no R-list configs to convert here. (The kuznets config is generated directly
# by tools/build_kuznets.py.) The convert_config helper above is kept for future use.
