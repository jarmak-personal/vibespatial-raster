"""Sphinx configuration for vibespatial-raster documentation."""

project = "vibespatial-raster"
copyright = "2026, vibespatial Contributors"
author = "vibespatial Contributors"
release = "0.1.6"

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
]

# -- sphinx-autoapi configuration -------------------------------------------
autoapi_dirs = ["../src/vibespatial"]
autoapi_type = "python"
autoapi_root = "api"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_ignore = [
    "**/kernels/*.py",
]
autoapi_keep_files = True
autoapi_add_toctree_entry = False
autoapi_python_use_implicit_namespaces = True
autoapi_member_order = "groupwise"

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
    "source_repository": "https://github.com/jarmak-personal/vibespatial-raster",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view", "edit"],
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/jarmak-personal/vibespatial-raster",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/vibespatial-raster/",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 17 20"><path d="M8.5 0L0 4.7v9.5L8.5 19l8.5-4.8V4.7L8.5 0zm0 2.2l5.9 3.3-5.9 3.3-5.9-3.3 5.9-3.3zM1.5 6l6.2 3.5v7L1.5 13V6z"></path></svg>',
            "class": "",
        },
    ],
    "light_css_variables": {},
    "dark_css_variables": {},
}

html_context = {
    "default_mode": "dark",
}
