#!/usr/bin/env Rscript
# Generate R reference ("golden") values from tests/fixtures/sample.csv.
#
# These use base-R functions that ExPanDaR relies on (sd with n-1, quantile type 7,
# cor.test with exact=FALSE) so the resulting goldens lock numerical parity with the
# original R package even when ExPanDaR itself is not installed.

suppressMessages(library(jsonlite))
here <- dirname(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]))
if (length(here) == 0) here <- "tests/fixtures"
df <- read.csv(file.path(here, "sample.csv"))

vars <- c("x1", "x2", "x3")
goldens <- list()

# Descriptive statistics
desc <- list()
for (v in vars) {
  x <- df[[v]]
  q <- stats::quantile(x, probs = c(0, .25, .5, .75, 1), na.rm = TRUE, type = 7)
  desc[[v]] <- list(
    N = sum(!is.na(x)), mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE),
    min = q[[1]], q25 = q[[2]], median = q[[3]], q75 = q[[4]], max = q[[5]]
  )
}
goldens$descriptive <- desc

# treat_outliers (winsorize at 5%, type 7)
wins <- list()
for (v in vars) {
  x <- df[[v]]
  lim <- stats::quantile(x, probs = c(.05, .95), na.rm = TRUE, type = 7)
  xw <- x
  xw[xw < lim[1]] <- lim[1]
  xw[xw > lim[2]] <- lim[2]
  wins[[v]] <- list(lo = lim[[1]], hi = lim[[2]], sum = sum(xw))
}
goldens$winsorize_p05 <- wins

# Correlations (Pearson and Spearman), exact = FALSE
corr <- list(pearson = list(), spearman = list())
pairs <- list(c("x1", "x2"), c("x1", "x3"), c("x2", "x3"))
for (m in c("pearson", "spearman")) {
  for (p in pairs) {
    ct <- suppressWarnings(stats::cor.test(df[[p[1]]], df[[p[2]]], method = m, exact = FALSE))
    corr[[m]][[paste0(p[1], "_", p[2])]] <- list(
      r = unname(ct$estimate), p = ct$p.value,
      n = sum(is.finite(df[[p[1]]]) & is.finite(df[[p[2]]]))
    )
  }
}
goldens$correlation <- corr

# Quantile by group (median of x3 by year)
med_by_year <- tapply(df$x3, df$year, function(z) stats::quantile(z, .5, type = 7, na.rm = TRUE))
goldens$median_x3_by_year <- as.list(med_by_year)

writeLines(jsonlite::toJSON(goldens, auto_unbox = TRUE, digits = 12, pretty = TRUE),
           file.path(here, "goldens.json"))
cat("wrote", file.path(here, "goldens.json"), "\n")
