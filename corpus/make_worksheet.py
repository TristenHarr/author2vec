#!/usr/bin/env python3
"""Dump every author from authors.toml into a worksheet so we can add new
style-shaping dimensions: where they were RAISED (learned to read and write) and
where they were EDUCATED (school / college), alongside their BIRTH country.

    python3 corpus/make_worksheet.py   ->  corpus/authors_worksheet.txt
"""
import re
import pathlib

HERE = pathlib.Path(__file__).parent
toml = (HERE / "authors.toml").read_text(encoding="utf-8")


def grab(block: str, key: str) -> str:
    m = re.search(rf'^\s*{key}\s*=\s*"([^"]*)"', block, re.M)
    return m.group(1) if m else ""


authors = []
for block in toml.split("[[authors]]")[1:]:
    name = grab(block, "name")
    if name:
        authors.append(
            {"name": name, "gender": grab(block, "gender"), "birth": grab(block, "country")}
        )

out = HERE / "authors_worksheet.txt"
with out.open("w", encoding="utf-8") as f:
    f.write(f"# author2vec dimensions worksheet — {len(authors)} authors\n")
    f.write("# RAISED   = where they grew up / learned to read and write\n")
    f.write("# EDUCATED = where they went to school / college\n")
    f.write("# (leave blank to mean 'same as birth country')\n\n")
    f.write(f"{'AUTHOR':26} | {'GENDER':6} | {'BIRTH':13} | {'RAISED':13} | EDUCATED\n")
    f.write("-" * 88 + "\n")
    for a in authors:
        f.write(f"{a['name']:26} | {a['gender']:6} | {a['birth']:13} | {'':13} | \n")

# also emit a country tally to see the current balance
from collections import Counter  # noqa: E402

tally = Counter(a["birth"] for a in authors)
print(f"wrote {len(authors)} authors -> {out}")
print("birth-country tally:", dict(sorted(tally.items(), key=lambda x: -x[1])))
