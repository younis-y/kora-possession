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
.venv/bin/python -m kora.run --out "Kora Analytics.csv"   # submission from the committed models (about 15 s)
shasum -a 256 -c models/submission.sha256                 # must print OK: identical to our submission
./build.sh                                                # everything from scratch: download, train, decode, check
.venv/bin/python -m kora.checks "Kora Analytics.csv"      # label-free sanity checks on the challenge output
```

`build.sh` runs on Linux and macOS. On macOS it uses `caffeinate` when available to keep the machine
awake during training. A submission hash mismatch fails the build with a nonzero exit status.

### Error metrics

```bash
.venv/bin/python -m kora.errors --predictions predictions.npz
.venv/bin/python -m unittest discover -s tests -v
```

Reads decoded predictions without training. The loader checks format and alignment, not whether
predictions came from clean evaluation. Without arguments, it uses legacy development OOF
predictions. See `.venv/bin/python -m kora.errors --help` for the bundle format and a synthetic example.

The match data is gated and never committed. `data/` and every derived array are gitignored.
See `docs/method.md` for the method and `CONTRIBUTING.md` for how we work.
