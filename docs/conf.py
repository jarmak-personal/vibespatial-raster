"""Sphinx configuration for vibespatial-raster documentation."""

project = "vibespatial-raster"
copyright = "2025, vibespatial Contributors"
author = "vibespatial Contributors"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "vibespatial": ("https://jarmak-personal.github.io/vibeSpatial/", None),
    "vibeproj": ("https://jarmak-personal.github.io/vibeProj/", None),
}

myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
    "html_admonition",
    "attrs_inline",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- Theme: Furo + vibeSpatial overlay ---------------------------------------
html_theme = "furo"
html_title = "vibespatial-raster"

html_static_path = ["_static"]
html_css_files = ["css/vibespatial.css"]
html_js_files = ["js/vibespatial.js"]

html_theme_options = {
    "source_repository": "https://github.com/picard/vibespatial-raster",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {},
    "dark_css_variables": {},
}

html_context = {
    "default_mode": "dark",
}
