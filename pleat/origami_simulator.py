"""Open a crease pattern in Origami Simulator (https://origamisimulator.org/).

Origami Simulator imports a crease pattern via a postMessage handshake: it
announces ``{from:'OrigamiSimulator', status:'ready'}`` to its parent frame (when
embedded) or opener (when popped out), then accepts
``{op:'importFold', fold:<FOLD object>}``. We embed the pattern as FOLD JSON in a
self-contained page and reply to whichever OS window reports ready.

Two entry points:

- :func:`origami_simulator` -- show OS folding a pattern: inline in the cell under
  Jupyter (Notebook / Lab / VS Code), or in the system browser from a script.
- :func:`origami_simulator_button` -- a button that embeds OS inline when clicked
  (lazy; good for the static docs, where auto-loading many at once would be heavy).
"""

from __future__ import annotations

import html
import json
import os
import tempfile
import uuid
import webbrowser

from .io.fold import graph_to_fold
from .utils import in_notebook

__all__ = ["origami_simulator", "origami_simulator_button"]

_OS_ORIGIN = "https://origamisimulator.org"
# The empty ``?model=`` query is load-bearing: it makes Origami Simulator skip
# loading its default demo (the waterbomb), which would otherwise finish loading
# *after* our importFold and clobber it. This mirrors erikdemaine.org's maze tool.
_OS_URL = _OS_ORIGIN + "/?model="


def _fold_json(G) -> str:
    """FOLD as a JSON string safe to embed inside an HTML ``<script>`` tag."""
    return json.dumps(graph_to_fold(G)).replace("</", "<\\/")


def _page_html(G) -> str:
    """A self-contained page: OS in a full-viewport iframe, importing *G* over the
    ready/importFold handshake, with a Fullscreen button."""
    fold_json = _fold_json(G)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>pleat &rarr; Origami Simulator</title>
<style>
  html,body{{margin:0;height:100%}}
  #os{{border:0;width:100vw;height:100vh;display:block}}
  #pop{{position:fixed;bottom:8px;right:8px;z-index:9;font:14px sans-serif;
       padding:6px 10px;cursor:pointer}}
</style></head><body>
<button id="pop">&#9974; Fullscreen</button>
<iframe id="os" src="{_OS_URL}"></iframe>
<script>
  const FOLD = {fold_json};
  const OS_ORIGIN = "{_OS_ORIGIN}";
  window.addEventListener('message', function(e){{
    // Only answer the real Origami Simulator (checked by origin, which the browser
    // sets and a page cannot forge), and hand the pattern only to that origin.
    if (e.origin === OS_ORIGIN && e.data && e.data.from === 'OrigamiSimulator' && e.data.status === 'ready')
      e.source.postMessage({{op:'importFold', fold: FOLD}}, OS_ORIGIN);
  }});
  document.getElementById('pop').addEventListener('click', function(){{
    // Enlarge via the Fullscreen API (Esc to exit). Unlike opening a popup, this
    // works inside a sandboxed webview (e.g. VS Code), where popups are suppressed.
    var el = document.documentElement;
    var req = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen;
    if (req) req.call(el);
  }});
</script></body></html>"""


def _iframe_html(G, *, height: int = 600) -> str:
    """``<iframe>`` markup embedding OS inline, with the page carried in ``srcdoc``.

    The handshake runs parent-to-child *within* the iframe, so there is no popup or
    opener link -- the part a sandboxed VS Code webview severs.
    """
    doc = html.escape(_page_html(G), quote=True)
    return (
        f'<iframe srcdoc="{doc}" '
        f'style="width:100%;height:{int(height)}px;border:1px solid #ccc" '
        f'allow="fullscreen"></iframe>'
    )


def _button_html(G, *, height: int = 600, title: str = "Load Origami Simulator") -> str:
    """A button that injects the OS iframe inline when clicked (no popup)."""
    payload = json.dumps(_iframe_html(G, height=height)).replace("</", "<\\/")
    uid = uuid.uuid4().hex[:8]
    return f"""<div id="pleat-os-{uid}">
<button style="font:14px sans-serif;padding:6px 10px;cursor:pointer">&#9654; {html.escape(title)}</button>
</div>
<script>
(function(){{
  var box = document.getElementById("pleat-os-{uid}");
  box.querySelector("button").addEventListener("click", function(){{ box.innerHTML = {payload}; }});
}})();
</script>"""


def _open_in_browser(G) -> str:
    """Write the page to a temp file and open it in the system browser."""
    fd, path = tempfile.mkstemp(prefix="pleat-os-", suffix=".html")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(_page_html(G))
    url = "file://" + path
    print(f"Opening Origami Simulator: {url}")
    webbrowser.open(url)
    return path


def _display_html(markup: str) -> None:
    """Display raw HTML inline. The raw mimebundle avoids IPython's ``HTML`` iframe
    warning and works off any line / several times per cell."""
    from IPython.display import display

    display({"text/html": markup}, raw=True)


def origami_simulator(G, *, height: int = 600, new_tab: bool = False) -> None:
    """Show Origami Simulator folding the crease pattern *G*.

    In a Jupyter environment (Notebook, Lab, VS Code) this embeds OS inline in the
    cell output (resizable via *height*); it can be called off any line and several
    times in one cell. From a plain script it opens OS in the system browser.

    Pass ``new_tab=True`` to force the browser even from a notebook - useful in VS
    Code, where the inline Fullscreen button is blocked by the webview.
    """
    if new_tab or not in_notebook():
        _open_in_browser(G)
    else:
        _display_html(_iframe_html(G, height=height))


def origami_simulator_button(G, *, height: int = 600, title: str = "Load Origami Simulator") -> None:
    """Display a button that embeds Origami Simulator inline when clicked.

    Like :func:`origami_simulator` but lazy -- nothing loads until the reader
    clicks, so a page with many patterns does not spin up a WebGL instance for each
    on load. *title* sets the button label.
    """
    _display_html(_button_html(G, height=height, title=title))
