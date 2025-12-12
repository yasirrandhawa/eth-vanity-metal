//! Precomputation table generation for fast scalar multiplication
//!
//! This module implements the profanity2 approach to EC scalar multiplication:
//! Instead of doing k*G via repeated point addition, we precompute a lookup table
//! that allows us to compute k*G using only ~32 point additions.
//!
//! ## Approach
//!
//! A 256-bit scalar k can be decomposed into 32 bytes:
//! k = k[0] + k[1]×256 + k[2]×256² + ... + k[31]×256³¹
//!
//! Therefore:
//! k×G = k[0]×G + k[1]×(256×G) + k[2]×(256²×G) + ... + k[31]×(256³¹×G)
//!
//! We precompute a table:
//! - For each byte position i ∈ [0, 31]
//! - For each byte value v ∈ [1, 255]
//! - Store: v × (256^i × G)
//!
//! This gives us 32 × 255 = 8160 precomputed points.
//! Total memory: 8160 × 64 bytes = ~522 KB
//!
//! ## Performance Impact
//!
//! - Traditional scalar multiplication: ~256 point operations (average)
//! - Precomputed approach: ~32 table lookups + 32 point additions
//! - Expected speedup: 5-10x on the EC math portion

use crate::gpu::native_search::GpuUint256;
use secp256k1::{Secp256k1, SecretKey, PublicKey};

/// Affine point representation (x, y only - smaller than Jacobian)
/// Must match Metal's AffinePoint struct exactly
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuAffinePoint {
    pub x: GpuUint256,
    pub y: GpuUint256,
}

impl GpuAffinePoint {
    /// Create from secp256k1 public key
    pub fn from_pubkey(pubkey: &PublicKey) -> Self {
        let serialized = pubkey.serialize_uncompressed();

        // Extract X and Y coordinates (skip 0x04 prefix)
        let x_bytes: [u8; 32] = serialized[1..33].try_into().unwrap();
        let y_bytes: [u8; 32] = serialized[33..65].try_into().unwrap();

        Self {
            x: GpuUint256::from(x_bytes),
            y: GpuUint256::from(y_bytes),
        }
    }
}

/// Generate the complete precomputation table for fast scalar multiplication
///
/// Returns a vector of 8160 affine points organized as:
/// [pos0_val1, pos0_val2, ..., pos0_val255, pos1_val1, ..., pos31_val255]
///
/// To look up the precomputed value for byte position i with value v:
/// index = i * 255 + (v - 1)
///
/// Note: We skip v=0 since 0×P = point at infinity (not needed)
pub fn generate_precomp_table() -> Vec<GpuAffinePoint> {
    let secp = Secp256k1::new();

    // 32 byte positions × 255 non-zero values
    let table_size = 32 * 255;
    let mut table = Vec::with_capacity(table_size);

    println!("  → Generating precomputation table ({} points)...", table_size);

    // For each byte position (0 to 31)
    for position in 0..32 {
        // For each byte value (1 to 255, skip 0)
        for value in 1u8..=255 {
            // Compute scalar: k = value × 256^position
            let mut scalar_bytes = [0u8; 32];
            scalar_bytes[position] = value;

            // Create secret key from scalar
            let secret_key = SecretKey::from_slice(&scalar_bytes)
                .expect("Valid scalar for precomputation");

            // Compute public key: P = k × G
            let public_key = PublicKey::from_secret_key(&secp, &secret_key);

            // Convert to affine point
            let point = GpuAffinePoint::from_pubkey(&public_key);
            table.push(point);
        }

        // Progress indicator every 8 positions
        if position % 8 == 7 {
            println!("    → Generated positions 0-{} ({}/{} points)",
                     position, table.len(), table_size);
        }
    }

    println!("  → Precomputation table complete ({} points, ~{} KB)",
             table.len(), (table.len() * 64) / 1024);

    assert_eq!(table.len(), table_size, "Precomputation table size mismatch");
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precomp_table_generation() {
        let table = generate_precomp_table();

        // Verify table size
        assert_eq!(table.len(), 32 * 255);

        // Verify we can access each entry
        for i in 0..32 {
            for v in 1u8..=255 {
                let index = i * 255 + (v as usize - 1);
                let point = &table[index];

                // Verify point is not zero (basic sanity check)
                let is_zero = point.x.d.iter().all(|&x| x == 0) &&
                             point.y.d.iter().all(|&y| y == 0);
                assert!(!is_zero, "Precomputed point should not be zero");
            }
        }
    }

    #[test]
    fn test_affine_point_from_pubkey() {
        let secp = Secp256k1::new();
        let secret = SecretKey::new(&mut rand::thread_rng());
        let pubkey = PublicKey::from_secret_key(&secp, &secret);

        let affine = GpuAffinePoint::from_pubkey(&pubkey);

        // Verify coordinates match
        let serialized = pubkey.serialize_uncompressed();
        let x_bytes = &serialized[1..33];
        let y_bytes = &serialized[33..65];

        assert_eq!(affine.x.to_bytes(), x_bytes);
        assert_eq!(affine.y.to_bytes(), y_bytes);
    }
}
