from importlib.metadata import entry_points

def get_plugins():
    try:
        eps = entry_points(group="dirigo.plugins")
    except TypeError:
        # Python <3.10 compatibility if needed
        eps = entry_points().get("dirigo.plugins", [])
    items = []
    for ep in sorted(eps, key=lambda e: e.name.lower()):
        pkg = ep.module.split(".")[0]
        items.append((ep.name, ep.value, pkg))
    return items

def write_markdown(path):
    items = get_plugins()
    if not items:
        content = "# Plugins\n\n_No plugins found at build time._\n"
    else:
        rows = [f"- **{name}** — `{value}` (package: `{pkg}`)" for name, value, pkg in items]
        content = "# Plugins\n\n" + "\n".join(rows) + "\n"
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
