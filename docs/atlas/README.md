# Atlas

`index.html` is the Brutmaschine Atlas: a page in German and English, switched at the top
right, with diagrams of the breeding loop, the Cascade, the judges, the overseers, the
soul and the character, a speckle simulator that runs in the browser, and the result of
step 1. Open it in a browser; it needs no server. Fonts come from Google Fonts, and
without a network the page falls back to system fonts.

The step 1 numbers sit in a JSON block between `<!-- step1-data:start -->` and
`<!-- step1-data:end -->`. `sync_step1.py` rewrites only that block from
`../step1-counted/analysis.txt`, recomputes every evidence value with exact fractions by
section 8 of `docs/PREREG_STEP1_CIRCUITS.md`, and stops with an error on any mismatch:

```sh
python3 -I -B docs/atlas/sync_step1.py docs/atlas/index.html
```
