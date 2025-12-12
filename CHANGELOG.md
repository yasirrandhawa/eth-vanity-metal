# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-13

### Added
- Initial release of eth-vanity-metal
- Metal GPU acceleration with 367 MH/s on M4 Pro
- Prefix and suffix pattern matching
- Precomputation table (8,160 EC points) for fast scalar multiplication
- 64-bit limb uint256 arithmetic for optimal GPU performance
- Montgomery batch inversion (window=32)
- Jacobian coordinate point addition
- Optimized Keccak-256 implementation
- CPU fallback mode for compatibility testing
- Real-time progress display with speed metrics
- Address validation against eth_account reference

### Performance
- **367 MH/s** on Apple M4 Pro (Metal GPU)
- **~85% faster** than profanity2 OpenCL (199 MH/s)
- **180x faster** than CPU-only generators

### Technical
- Full EC math (secp256k1) on GPU
- 131,072 GPU threads × 2,048 steps per batch
- Fast modular reduction using secp256k1 prime structure
- Fixed carry detection bug in schoolbook multiplication

## [Unreleased]

### Planned
- GLV endomorphism optimization
- Multi-GPU support
- Batch-32 parallelism
- EIP-55 checksum pattern support
