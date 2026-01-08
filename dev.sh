#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
WASM_DIR="$PROJECT_ROOT/wasm-lib"
WEBAPP_DIR="$PROJECT_ROOT/webapp"

echo "🧹 Cleaning up old processes and caches..."

# Kill any running dev servers on port 5173
if lsof -ti:5173 >/dev/null 2>&1; then
    echo "  Killing old dev servers..."
    lsof -ti:5173 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Clean Vite cache
echo "  Clearing Vite cache..."
rm -rf "$WEBAPP_DIR/node_modules/.vite"
rm -rf "$WEBAPP_DIR/.svelte-kit"

echo ""
echo "🦀 Building WASM..."
cd "$WASM_DIR"

# Clean and rebuild WASM to default location (wasm-lib/pkg)
rm -rf pkg
wasm-pack build --target web

echo ""
echo "✨ Starting dev server..."
cd "$WEBAPP_DIR"
npm run dev
