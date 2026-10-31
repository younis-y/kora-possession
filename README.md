# kora-possession

Kora Analytics' entry for [More Than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027):
one label per frame of the tactical video (H / A / D / X) for FC Barcelona vs Athletic Club, from the
automatic tracking, homography and smart tagging supplied by the organisers.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# unpack the match bundle into ./data/{tactical,broadcast}/ or export MTAH_DATA_DIR=/path/to/bundle
```

## Run

```bash
.venv/bin/python -m kora.data               # parse the bundle once into data/cache (about 25 s)
.venv/bin/python -m kora.profile            # descriptive profile of the data, markdown to stdout
.venv/bin/python -m kora.run --out "Kora Analytics.csv"   # the submission (coming)
```

The match data is gated and never committed; `data/` and every derived array are gitignored.
