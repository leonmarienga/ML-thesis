"""Verify structural navigation and accessible names without claiming browser/AT QA."""
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

class Structure(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.labels = set()
        self.selects = set()
        self.targets = []
        self.headers = 0
        self.landmarks = []
        self.captions = 0

    def handle_starttag(self, tag, attrs):
        kv = dict(attrs)
        if "id" in kv:
            self.ids.append(kv["id"])
        if tag == "label" and "for" in kv:
            self.labels.add(kv["for"])
        if tag == "select":
            self.selects.add(kv["id"])
        if kv.get("href","").startswith("#"):
            self.targets.append(kv["href"][1:])
        if "aria-labelledby" in kv:
            self.targets.extend(kv["aria-labelledby"].split())
        if tag=="th":
            assert kv.get("scope")=="col"
            self.headers += 1
        if tag in ("main","nav"):
            self.landmarks.append(tag)
        if tag=="caption":
            self.captions += 1

p = Structure()
p.feed(Path(__file__).with_name("index.html").read_text())
assert len(p.ids)==len(set(p.ids))
assert set(p.targets).issubset(set(p.ids))
assert p.labels == {"interval-level","veto-primary","veto-other","veto-regime"}
assert p.labels.issubset(p.selects)
assert Counter(p.landmarks)=={"main":1,"nav":1}
assert p.captions==3 and p.headers==12
print("PASS: unique IDs, all navigation and aria-labelledby targets valid, core select labels, main/navigation landmarks, captions and column scopes.")
