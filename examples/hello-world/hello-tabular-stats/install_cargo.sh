
# fastdigest (or its dependencies) requires Rust and Cargo to build. 
# You need to install Rust and Cargo on your Ubuntu system. Follow these steps:
# Install Rust and Cargo
# Run the following command to install Rust using rustup:


curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Then restart your terminal or run:

source $HOME/.cargo/env
# Verify Installation
# Check if Rust and Cargo are installed correctly:
rustc --version
cargo --version