#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
WASM_DIR="$PROJECT_ROOT/wasm-lib"
WEBAPP_DIR="$PROJECT_ROOT/webapp"

echo "🦀 Building WASM (release mode)..."
cd "$WASM_DIR"
rm -rf pkg
wasm-pack build --target web --release

echo ""
echo "📦 Building webapp..."
cd "$WEBAPP_DIR"
npm run build

echo ""
echo "✅ Production build complete!"
echo "Output: $WEBAPP_DIR/build"
