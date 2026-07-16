"""
Fix Python 3.14 SyntaxError in Trading bot notebook.ipynb.

Problem: Python 3.14 tokenizer rejects '1h' (e.g. --timeframe 1h) as an
invalid decimal literal, even inside %%bash magic cells.

Fix: Convert the %%bash cell to per-line ! IPython shell escapes.
IPython routes !-prefixed lines directly to the shell without Python-tokenizing
the arguments, so '1h' is never seen by Python's parser.
"""
import json

notebook_path = "Trading bot notebook.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

changed = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    source = cell.get("source", [])
    # Target: the %%bash cell that runs update_market_data with --timeframe 1h
    if source and source[0].startswith("%%bash") and any(
        "update_market_data" in line for line in source
    ):
        new_source = []
        for line in source[1:]:  # skip the %%bash line itself
            stripped = line.rstrip("\n")
            if stripped == "" or stripped.startswith("#"):
                # Keep blank lines and comments as-is
                new_source.append(line)
            else:
                # Prefix bare shell commands with !
                nl = "\n" if line.endswith("\n") else ""
                new_source.append("!" + stripped + nl)
        cell["source"] = new_source
        changed = True
        print("✅ Converted %%bash cell to per-line ! shell escapes.")
        break

if not changed:
    print("ℹ️  No %%bash cell found to convert (already fixed or not present).")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("💾 Notebook saved.")
