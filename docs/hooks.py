"""mkdocs hooks (wired via `hooks:` in mkdocs.yml)."""

import re


def on_page_content(html, page, config, files):
    # mkdocs-jupyter passes markdown-cell links through verbatim, so relative
    # `.ipynb` cross-links (which work in Jupyter and on GitHub) would 404 on
    # the built site. Rewrite them to the rendered `.html` pages.
    return re.sub(r'href="(?!https?://)([^"]+)\.ipynb"', r'href="\1.html"', html)
