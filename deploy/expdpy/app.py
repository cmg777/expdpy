"""shinyapps.io entry point — expdpy Shiny app in upload (bring-your-own-data) mode.

shinyapps.io / ``shiny run`` serve the module-level ``app`` object below. ``ExPdPy(run=False)``
builds and returns the ``shiny.App`` without calling ``.run()``; with no dataframe it opens the
upload dialog so visitors supply their own data.
"""

from expdpy.app import ExPdPy

app = ExPdPy(run=False)
