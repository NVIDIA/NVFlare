# Rust 1.95.0, same build base as the pinned Trustee Attestation Service.
FROM docker.io/library/rust@sha256:f49565f188ee00bc2a18dd418183f2c5f23ef7d6e691890517ed341a598f67c3

RUN apt-get update && apt-get install --no-install-recommends -y \
        ca-certificates clang curl gnupg libssl-dev pkg-config

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY verify-intel-key.sh ./verify-intel-key.sh
RUN curl -fsSL https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key \
        -o /tmp/intel-sgx-deb.key && \
    bash ./verify-intel-key.sh /tmp/intel-sgx-deb.key /usr/share/keyrings/intel-sgx.gpg && \
    printf '%s\n' 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-sgx.gpg] https://download.01.org/intel-sgx/sgx_repo/ubuntu noble main' \
        > /etc/apt/sources.list.d/intel-sgx.list && \
    apt-get update && apt-get install --no-install-recommends -y \
        libsgx-dcap-quote-verify-dev=1.27.100.1-noble1 libsgx-dcap-quote-verify=1.27.100.1-noble1

RUN cargo check --locked --tests && cargo test --locked && cargo build --release --locked && \
    install -m 0755 target/release/tdx-evidence-verify /usr/local/bin/tdx-evidence-verify

# Keep the build/runtime ABI identical, including Intel's QVL. No SGX/TDX
# device access is required for QVL-based verification on the trusted host.
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/tdx-evidence-verify"]
