#!/usr/bin/env python3
"""Rewrite authors.toml: set `country` to the LITERAL birthplace and add
`raised` (where they grew up / learned to read & write), `educated` (country of their
main education), and `college` (specific university, or "None" if self-taught).
Every author's book list is preserved verbatim.

    python3 corpus/apply_dimensions.py

Values are (birth, raised, educated_country, college). Debatable calls are flagged.
"""
import re
import pathlib
from collections import Counter

DIMS = {
    "Jane Austen": ("England", "England", "England", "None"),
    "Charles Dickens": ("England", "England", "England", "None"),
    "Mark Twain": ("United States", "United States", "United States", "None"),
    "Edgar Allan Poe": ("United States", "United States", "United States", "University of Virginia"),
    "Arthur Conan Doyle": ("Scotland", "Scotland", "Scotland", "University of Edinburgh"),
    "Mary Shelley": ("England", "England", "England", "None"),
    "Oscar Wilde": ("Ireland", "Ireland", "England", "Oxford"),
    "Herman Melville": ("United States", "United States", "United States", "None"),
    "Bram Stoker": ("Ireland", "Ireland", "Ireland", "Trinity College Dublin"),
    "Louisa May Alcott": ("United States", "United States", "United States", "None"),
    "H. G. Wells": ("England", "England", "England", "Royal College of Science"),
    "Charlotte Bronte": ("England", "England", "England", "None"),
    "Lewis Carroll": ("England", "England", "England", "Oxford"),
    "Robert Louis Stevenson": ("Scotland", "Scotland", "Scotland", "University of Edinburgh"),
    "George Eliot": ("England", "England", "England", "None"),
    "Anne Bronte": ("England", "England", "England", "None"),
    "Elizabeth Gaskell": ("England", "England", "England", "None"),
    "Edith Wharton": ("United States", "United States", "United States", "None"),
    "Willa Cather": ("United States", "United States", "United States", "University of Nebraska"),
    "Nathaniel Hawthorne": ("United States", "United States", "United States", "Bowdoin College"),
    "Jonathan Swift": ("Ireland", "Ireland", "Ireland", "Trinity College Dublin"),
    "Walter Scott": ("Scotland", "Scotland", "Scotland", "University of Edinburgh"),
    "L. M. Montgomery": ("Canada", "Canada", "Canada", "Dalhousie University"),
    "Stephen Leacock": ("England", "Canada", "Canada", "University of Toronto"),
    "Miles Franklin": ("Australia", "Australia", "Australia", "None"),
    "Marcus Clarke": ("England", "England", "England", "None"),
    "Jack London": ("United States", "United States", "United States", "UC Berkeley"),
    "Stephen Crane": ("United States", "United States", "United States", "Syracuse University"),
    "Kate Chopin": ("United States", "United States", "United States", "None"),
    "Harriet Beecher Stowe": ("United States", "United States", "United States", "None"),
    "Sarah Orne Jewett": ("United States", "United States", "United States", "None"),
    "George MacDonald": ("Scotland", "Scotland", "Scotland", "University of Aberdeen"),
    "Margaret Oliphant": ("Scotland", "Scotland", "Scotland", "None"),
    "Maria Edgeworth": ("England", "Ireland", "England", "None"),
    "Sheridan Le Fanu": ("Ireland", "Ireland", "Ireland", "Trinity College Dublin"),
    "Oliver Goldsmith": ("Ireland", "Ireland", "Ireland", "Trinity College Dublin"),
    "Susanna Moodie": ("England", "England", "England", "None"),
    "Sara Jeannette Duncan": ("Canada", "Canada", "Canada", "None"),
    "Rolf Boldrewood": ("England", "Australia", "Australia", "None"),
    "Ethel Turner": ("England", "Australia", "Australia", "None"),
    "Arthur Machen": ("Wales", "Wales", "England", "None"),
    "Olive Schreiner": ("South Africa", "South Africa", "South Africa", "None"),
    "Rudyard Kipling": ("India", "England", "England", "None"),
    "Allen Raine": ("Wales", "Wales", "England", "None"),
    "W. H. Davies": ("Wales", "Wales", "Wales", "None"),
    "Percy FitzPatrick": ("South Africa", "South Africa", "South Africa", "None"),
    "Bertram Mitford": ("England", "England", "England", "None"),
    "Toru Dutt": ("India", "India", "England", "None"),
    "Cornelia Sorabji": ("India", "India", "England", "Oxford"),
    "Katherine Mansfield": ("New Zealand", "New Zealand", "England", "None"),
    "Ralph Connor": ("Canada", "Canada", "Canada", "University of Toronto"),
    "Catharine Parr Traill": ("England", "England", "England", "None"),
    "Ada Cambridge": ("England", "England", "England", "None"),
    "Joseph Furphy": ("Australia", "Australia", "Australia", "None"),
    "John Buchan": ("Scotland", "Scotland", "England", "Oxford"),
}


def grab(block, key):
    m = re.search(rf'^\s*{key}\s*=\s*"([^"]*)"', block, re.M)
    return m.group(1) if m else ""


HERE = pathlib.Path(__file__).parent
path = HERE / "authors.toml"
parts = path.read_text(encoding="utf-8").split("[[authors]]")
out = [parts[0]]
missing = []
for block in parts[1:]:
    name = grab(block, "name")
    if name not in DIMS:
        missing.append(name)
        out.append("[[authors]]" + block)
        continue
    birth, raised, edu, college = DIMS[name]
    block = re.sub(r'(?m)^\s*(raised|educated|college)\s*=\s*"[^"]*"\n', "", block)  # idempotent
    block = re.sub(
        r'(?m)^(\s*)country\s*=\s*"[^"]*"',
        lambda m: (
            f'{m.group(1)}country = "{birth}"\n'
            f'{m.group(1)}raised = "{raised}"\n'
            f'{m.group(1)}educated = "{edu}"\n'
            f'{m.group(1)}college = "{college}"'
        ),
        block,
        count=1,
    )
    out.append("[[authors]]" + block)

path.write_text("".join(out), encoding="utf-8")

for label, idx in [("birth", 0), ("raised", 1), ("educated", 2), ("college", 3)]:
    tally = Counter(v[idx] for v in DIMS.values())
    print(f"{label:9}:", dict(sorted(tally.items(), key=lambda x: -x[1])))
print("missing from DIMS (should be empty):", missing)
