#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMSDK_DIR="$SCRIPT_DIR/emsdk"
BUILD_DIR="$SCRIPT_DIR/wasm-cpp/build"
PKG_DIR="$SCRIPT_DIR/wasm-cpp/pkg"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_step() {
    echo -e "${GREEN}==>${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

echo_error() {
    echo -e "${RED}Error:${NC} $1"
}

# Check and install Emscripten SDK if needed
install_emsdk() {
    if [ ! -d "$EMSDK_DIR" ]; then
        echo_step "Installing Emscripten SDK..."
        git clone https://github.com/emscripten-core/emsdk.git "$EMSDK_DIR"
        cd "$EMSDK_DIR"
        ./emsdk install latest
        ./emsdk activate latest
        cd "$SCRIPT_DIR"
    else
        echo_step "Emscripten SDK found at $EMSDK_DIR"
    fi
}

# Source Emscripten environment
source_emsdk() {
    echo_step "Sourcing Emscripten environment..."
    source "$EMSDK_DIR/emsdk_env.sh"
}

# Build the C++ WASM module
build_wasm() {
    echo_step "Building C++ WASM module..."

    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Configure with Emscripten
    emcmake cmake .. -DCMAKE_BUILD_TYPE=Release

    # Build
    emmake make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

    cd "$SCRIPT_DIR"
}

# Create output package directory
create_pkg() {
    echo_step "Creating package..."
    mkdir -p "$PKG_DIR"

    # Copy TypeScript definitions if they exist
    if [ -f "$SCRIPT_DIR/wasm-cpp/src/drone_swarm.d.ts" ]; then
        cp "$SCRIPT_DIR/wasm-cpp/src/drone_swarm.d.ts" "$PKG_DIR/"
    fi

    echo_step "Build complete! Output in $PKG_DIR"
    ls -la "$PKG_DIR"
}

# Clean build
clean() {
    echo_step "Cleaning build directories..."
    rm -rf "$BUILD_DIR"
    rm -rf "$PKG_DIR"
    echo_step "Clean complete"
}

# Main
case "${1:-build}" in
    install)
        install_emsdk
        source_emsdk
        ;;
    build)
        install_emsdk
        source_emsdk
        build_wasm
        create_pkg
        ;;
    clean)
        clean
        ;;
    rebuild)
        clean
        install_emsdk
        source_emsdk
        build_wasm
        create_pkg
        ;;
    *)
        echo "Usage: $0 {install|build|clean|rebuild}"
        echo ""
        echo "Commands:"
        echo "  install  - Install Emscripten SDK only"
        echo "  build    - Build WASM module (default)"
        echo "  clean    - Remove build artifacts"
        echo "  rebuild  - Clean and rebuild"
        exit 1
        ;;
esac
