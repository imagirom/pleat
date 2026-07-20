/* Collapse the API Reference section in the sidebar on first load.
   navigation.expand expands every section via the md-toggle--indeterminate
   class; this strips it (and unchecks) for the API Reference subtree only,
   so the extensive module tree starts collapsed while notebook sections
   stay expanded. Clicking still toggles normally. */
for (const item of document.querySelectorAll(".md-nav--primary > .md-nav__list > .md-nav__item")) {
  const label = item.querySelector(".md-nav__link");
  if (!label || label.textContent.trim() !== "API Reference") continue;
  for (const toggle of item.querySelectorAll(".md-nav__toggle")) {
    if (toggle.hasAttribute("checked")) continue; // keep the active trail open when on an API page
    toggle.indeterminate = false;
    toggle.checked = false;
    toggle.classList.remove("md-toggle--indeterminate");
  }
}
