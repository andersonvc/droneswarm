#!/bin/bash
# Deploy the best trained model to the webapp for WASM inference.
# Fixes NaN values and ensures WASM-compatible normalizer format.

set -e

SRC_DIR="results"
DST_DIR="webapp/static/models"

# Find the best model (or use argument)
MODEL="${1:-$SRC_DIR/best_model.json}"

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    echo "Usage: ./deploy_model.sh [path/to/model.json]"
    exit 1
fi

echo "Deploying: $MODEL"

# Copy model weights
cp "$MODEL" "$DST_DIR/best_model.json"
echo "  Copied model ($(wc -c < "$DST_DIR/best_model.json" | tr -d ' ') bytes)"

# Fix normalizer: replace NaN/Infinity, ensure 'entity' key exists
NORM_SRC="${MODEL%.json}_normalizers.json"
if [ -f "$NORM_SRC" ]; then
    python3 -c "
import json, math

with open('$NORM_SRC') as f:
    s = f.read()

# Replace invalid JSON values
s = s.replace('NaN', '0.0').replace('Infinity', '1.0').replace('-Infinity', '0.0')
d = json.loads(s)

# Ensure ego has skip_indices
if 'ego' in d and 'skip_indices' not in d['ego']:
    d['ego']['skip_indices'] = []

# Ensure entity key exists (WASM expects it even though normalization is disabled)
if 'entity' not in d:
    d['entity'] = {'mean': [0.0]*10, 'var': [1.0]*10, 'skip_indices': [6, 7, 9]}

# Sanitize value normalizer
if 'value' in d:
    for k in ['mean', 'var']:
        v = d['value'].get(k, 0.0)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            d['value'][k] = 0.0 if k == 'mean' else 1.0

with open('$DST_DIR/best_model_normalizers.json', 'w') as f:
    json.dump(d, f)
"
    echo "  Fixed and copied normalizer"
else
    # No normalizer file — create a default one
    python3 -c "
import json
d = {
    'ego': {'mean': [0.0]*25, 'var': [1.0]*25, 'skip_indices': []},
    'entity': {'mean': [0.0]*10, 'var': [1.0]*10, 'skip_indices': [6, 7, 9]}
}
with open('$DST_DIR/best_model_normalizers.json', 'w') as f:
    json.dump(d, f)
"
    echo "  Created default normalizer (no source found)"
fi

# Verify
python3 -c "
import json
with open('$DST_DIR/best_model.json') as f:
    d = json.load(f)
has_attn2 = 'attn2' in d
with open('$DST_DIR/best_model_normalizers.json') as f:
    s = f.read()
has_nan = 'NaN' in s
n = json.loads(s)
print(f'  Model: {len(d)} keys, attn2={has_attn2}')
print(f'  Normalizer: {list(n.keys())}, NaN={has_nan}')
if not has_attn2:
    print('  WARNING: model missing attn2 — was it trained with 2-layer attention?')
if has_nan:
    print('  WARNING: normalizer still has NaN!')
"

echo "Done. Rebuild webapp to deploy."
