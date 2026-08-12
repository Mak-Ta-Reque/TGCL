# Fixed bugs (historical record)

## `inflect` mis-singularizing "bus" (and similar words) into garbage

**Fixed:** PR #1 (`preprocessing/inflect_utils.py`, merged as `91968f9`).

`inflect.engine().singular_noun("bus")` returns `"bu"` -- it treats the
trailing "s" as a plural marker even though "bus" is already singular (same
misfire on "glass"->"glas", "class"->"clas"). Both
`preprocessing/concept_image_mapping.py` and `preprocessing/select_top_concepts.py`
called `p.singular_noun(w) or w` unconditionally, so every AU-AIR "bus"
concept was silently renamed to "bu" throughout the pipeline (crop
filenames, feature files, concept-bank entries). Fixed with a denylist-guarded
`safe_singular_noun()` helper, covered by `tests/test_inflect_utils.py`.

## `tgcl` (then `cgdl`) contrastive prompt not forcing the exact concept word

**Fixed:** PR #1 (`src/models/constants.py`, merged as `b1e1dd3`).

The contrastive template ("Classify the image as either `[concept]` or No
`[concept]`...") hooks the position where the model *generates the tag word
itself*. For AU-AIR's `motorbike`, the model consistently answered with its
preferred synonym "motorcycle" instead, so 0 samples ever matched the
token-of-interest filter and the concept got zero directions in the
decomposition -- confirmed directly in the pipeline log:
`Skipping decomposition for '...motorbike.pth': 0 samples survived
token-of-interest filtering (tag word never found in this tag's
responses)`. Fixed by adding "Use exactly the word `[concept]` -- not a
synonym -- in your answer" to the template.
