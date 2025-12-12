//! GPU-Native Vanity Search for Ethereum
//! Runs the full EC math + Keccak on GPU for maximum speed

use crate::gpu::{GpuError, MetalContext};
use secp256k1::{SecretKey, Secp256k1, Scalar};
use std::sync::Arc;
use std::io::Write;

// ==========================================
// GPU-Compatible Structs
// ==========================================

/// Must match uint256_t in Metal (4 x u64, little-endian)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuUint256 {
    pub d: [u64; 4],  // Changed from [u32; 8] to match Metal's ulong d[4]
}

/// Must match JacobianPoint in Metal
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuJacobianPoint {
    pub x: GpuUint256,
    pub y: GpuUint256,
    pub z: GpuUint256,
}

impl From<[u8; 32]> for GpuUint256 {
    fn from(bytes: [u8; 32]) -> Self {
        let mut d = [0u64; 4];
        // Convert big-endian bytes to little-endian 64-bit limbs
        // d[0] is LSB limb (bytes[24..31]), d[3] is MSB limb (bytes[0..7])
        for i in 0..4 {
            let start = (3 - i) * 8;  // d[0] <- bytes[24..31], d[3] <- bytes[0..7]
            d[i] = u64::from_be_bytes([
                bytes[start], bytes[start+1], bytes[start+2], bytes[start+3],
                bytes[start+4], bytes[start+5], bytes[start+6], bytes[start+7]
            ]);
        }
        Self { d }
    }
}

impl GpuUint256 {
    pub fn one() -> Self {
        let mut d = [0u64; 4];
        d[0] = 1;  // LSB limb
        Self { d }
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        // d[3] is MSB -> bytes[0..7]
        // d[0] is LSB -> bytes[24..31]
        for i in 0..4 {
            let limb = self.d[3 - i];  // Start from MSB
            let start = i * 8;
            bytes[start..start+8].copy_from_slice(&limb.to_be_bytes());
        }
        bytes
    }
}

// ==========================================
// Hex Pattern Parsing
// ==========================================

/// Parse hex pattern string to bytes
/// Example: "dead" → [0xDE, 0xAD]
pub fn parse_hex_pattern(pattern: &str) -> Result<Vec<u8>, String> {
    if pattern.len() % 2 != 0 {
        return Err(format!(
            "Hex pattern '{}' must have even length (full bytes). Use '{}0' or '0{}' to match '{}'",
            pattern, pattern, pattern, pattern
        ));
    }

    hex::decode(pattern).map_err(|e| format!("Invalid hex pattern: {}", e))
}

// ==========================================
// Seed Generation
// ==========================================

/// Generate GPU seeds - starting points spread across search space
/// Uses GPU precomputation table for parallel public key generation
pub fn generate_gpu_seeds(
    searcher: &GpuNativeSearcher,
    num_threads: usize,
    steps_per_thread: u64,
) -> Result<(Vec<GpuJacobianPoint>, Vec<GpuUint256>, SecretKey), GpuError> {
    // Generate random base private key
    let base_key = SecretKey::new(&mut rand::thread_rng());

    let mut privkeys = Vec::with_capacity(num_threads);

    // Offset between threads
    let offset_bytes = steps_per_thread.to_be_bytes();
    let mut offset_32 = [0u8; 32];
    offset_32[24..32].copy_from_slice(&offset_bytes);
    let offset_scalar = Scalar::from_be_bytes(offset_32)
        .map_err(|_| GpuError::InitializationFailed("Invalid scalar".to_string()))?;

    let mut current_key = base_key;

    // Generate private keys for each thread
    for _ in 0..num_threads {
        // Store private key for result recovery
        let priv_bytes: [u8; 32] = current_key.secret_bytes();
        privkeys.push(GpuUint256::from(priv_bytes));

        // Advance to next thread's starting position
        current_key = current_key.add_tweak(&offset_scalar)
            .map_err(|_| GpuError::InitializationFailed("Key overflow".to_string()))?;
    }

    // Compute public keys on GPU using precomputation table
    // This parallelizes the expensive scalar multiplication
    let points = searcher.generate_seeds_gpu(&privkeys)?;

    Ok((points, privkeys, base_key))
}

// ==========================================
// GPU Search Execution
// ==========================================

pub struct GpuNativeSearcher {
    context: Arc<MetalContext>,
    pipeline: metal::ComputePipelineState,
    seed_pipeline: metal::ComputePipelineState,
    num_threads: usize,
    steps_per_thread: u32,
    precomp_buffer: metal::Buffer,
}

impl GpuNativeSearcher {
    pub fn new(context: Arc<MetalContext>, num_threads: usize, steps_per_thread: u32) -> Result<Self, GpuError> {
        // Load and compile the search_native.metal shader
        let shader_source = include_str!("search_native.metal");

        println!("  → Loading shader source ({} bytes)", shader_source.len());
        std::io::stdout().flush().unwrap();

        println!("  → Compiling Metal shader (this may take 10-30 seconds)...");
        std::io::stdout().flush().unwrap();

        // Create compile options with fast-math disabled to speed up compilation
        let compile_options = metal::CompileOptions::new();
        // Note: We can't set many options through the Rust bindings, but compilation should still work

        let library = context.device()
            .new_library_with_source(shader_source, &compile_options)
            .map_err(|e| GpuError::ShaderCompilationFailed(e.to_string()))?;

        println!("  → Shader compiled successfully");
        std::io::stdout().flush().unwrap();

        println!("  → Getting kernel function...");
        std::io::stdout().flush().unwrap();

        let function = library.get_function("eth_vanity_search", None)
            .map_err(|e| GpuError::ShaderCompilationFailed(e.to_string()))?;

        println!("  → Creating compute pipeline...");
        std::io::stdout().flush().unwrap();

        let pipeline = context.device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| GpuError::PipelineCreationFailed(e.to_string()))?;

        println!("  → Pipeline created successfully");
        std::io::stdout().flush().unwrap();

        println!("  → Getting seed generation kernel function...");
        std::io::stdout().flush().unwrap();

        let seed_function = library.get_function("generate_seeds", None)
            .map_err(|e| GpuError::ShaderCompilationFailed(e.to_string()))?;

        println!("  → Creating seed generation pipeline...");
        std::io::stdout().flush().unwrap();

        let seed_pipeline = context.device()
            .new_compute_pipeline_state_with_function(&seed_function)
            .map_err(|e| GpuError::PipelineCreationFailed(e.to_string()))?;

        println!("  → Seed generation pipeline created successfully");
        std::io::stdout().flush().unwrap();

        // Generate precomputation table for fast scalar multiplication
        println!("  → Generating precomputation table...");
        std::io::stdout().flush().unwrap();

        let precomp_table = crate::gpu::generate_precomp_table();

        // Create GPU buffer for precomputation table
        // Using StorageModeShared for compatibility, marked as constant in kernel
        let precomp_buffer_size = (precomp_table.len() * std::mem::size_of::<crate::gpu::GpuAffinePoint>()) as u64;
        let precomp_buffer = context.device().new_buffer_with_data(
            precomp_table.as_ptr() as *const _,
            precomp_buffer_size,
            metal::MTLResourceOptions::StorageModeShared,
        );

        println!("  → Precomputation table uploaded to GPU ({} KB)",
                 precomp_buffer_size / 1024);
        std::io::stdout().flush().unwrap();

        Ok(Self {
            context,
            pipeline,
            seed_pipeline,
            num_threads,
            steps_per_thread,
            precomp_buffer,
        })
    }

    /// Generate starting points from private keys using GPU precomputation table
    /// This parallelizes the expensive scalar multiplication
    pub fn generate_seeds_gpu(
        &self,
        privkeys: &[GpuUint256],
    ) -> Result<Vec<GpuJacobianPoint>, GpuError> {
        let device = self.context.device();

        // Create buffer for private keys
        let privkeys_buffer = device.new_buffer_with_data(
            privkeys.as_ptr() as *const _,
            (privkeys.len() * std::mem::size_of::<GpuUint256>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create buffer for output points
        let output_size = (privkeys.len() * std::mem::size_of::<GpuJacobianPoint>()) as u64;
        let output_buffer = device.new_buffer(
            output_size,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer
        let command_buffer = self.context.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline and buffers
        encoder.set_compute_pipeline_state(&self.seed_pipeline);
        encoder.set_buffer(0, Some(&privkeys_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&self.precomp_buffer), 0);

        // Dispatch threads (one per private key)
        let threadgroup_size = 256.min(self.seed_pipeline.max_total_threads_per_threadgroup()) as u64;
        let threadgroups = (privkeys.len() as u64 + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let output_ptr = output_buffer.contents() as *const GpuJacobianPoint;
        let output_slice = unsafe {
            std::slice::from_raw_parts(output_ptr, privkeys.len())
        };

        Ok(output_slice.to_vec())
    }

    /// Run a single search iteration
    /// Returns (found, thread_id, offset) if match found
    pub fn search_iteration(
        &self,
        points: &[GpuJacobianPoint],
        privkeys: &[GpuUint256],
        pattern: &[u8],
        is_suffix: bool,
    ) -> Result<Option<(u32, u32)>, GpuError> {
        let device = self.context.device();

        // Create buffers
        let points_buffer = device.new_buffer_with_data(
            points.as_ptr() as *const _,
            (points.len() * std::mem::size_of::<GpuJacobianPoint>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let privkeys_buffer = device.new_buffer_with_data(
            privkeys.as_ptr() as *const _,
            (privkeys.len() * std::mem::size_of::<GpuUint256>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let pattern_buffer = device.new_buffer_with_data(
            pattern.as_ptr() as *const _,
            pattern.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let pattern_len: u32 = pattern.len() as u32;
        let pattern_len_buffer = device.new_buffer_with_data(
            &pattern_len as *const _ as *const _,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let is_suffix_u32: u32 = if is_suffix { 1 } else { 0 };
        let is_suffix_buffer = device.new_buffer_with_data(
            &is_suffix_u32 as *const _ as *const _,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Result buffers
        let found_flag: u32 = 0;
        let found_buffer = device.new_buffer_with_data(
            &found_flag as *const _ as *const _,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let result_thread: u32 = 0;
        let result_thread_buffer = device.new_buffer_with_data(
            &result_thread as *const _ as *const _,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let result_offset: u32 = 0;
        let result_offset_buffer = device.new_buffer_with_data(
            &result_offset as *const _ as *const _,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let steps_buffer = device.new_buffer_with_data(
            &self.steps_per_thread as *const _ as *const _,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer
        let command_buffer = self.context.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline and buffers
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&points_buffer), 0);
        encoder.set_buffer(1, Some(&privkeys_buffer), 0);
        encoder.set_buffer(2, Some(&pattern_buffer), 0);
        encoder.set_buffer(3, Some(&pattern_len_buffer), 0);
        encoder.set_buffer(4, Some(&is_suffix_buffer), 0);
        encoder.set_buffer(5, Some(&found_buffer), 0);
        encoder.set_buffer(6, Some(&result_thread_buffer), 0);
        encoder.set_buffer(7, Some(&result_offset_buffer), 0);
        encoder.set_buffer(8, Some(&steps_buffer), 0);
        // Note: precomp_buffer is only used by generate_seeds kernel, not this kernel

        // Dispatch threads
        let threadgroup_size = 256.min(self.pipeline.max_total_threads_per_threadgroup()) as u64;
        let threadgroups = (self.num_threads as u64 + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let found_ptr = found_buffer.contents() as *const u32;
        let thread_ptr = result_thread_buffer.contents() as *const u32;
        let offset_ptr = result_offset_buffer.contents() as *const u32;

        let found = unsafe { *found_ptr };
        if found > 0 {
            let thread_id = unsafe { *thread_ptr };
            let offset = unsafe { *offset_ptr };
            Ok(Some((thread_id, offset)))
        } else {
            Ok(None)
        }
    }
}

/// Recover private key from GPU search result
pub fn recover_private_key(
    base_key: &SecretKey,
    thread_id: u32,
    offset: u32,
    steps_per_thread: u64,
) -> Result<SecretKey, String> {
    let _secp = Secp256k1::new();
    
    // Calculate total offset: thread_id * steps_per_thread + offset
    let thread_offset = (thread_id as u64)
        .checked_mul(steps_per_thread)
        .ok_or("Thread offset overflow")?;
    
    let total_offset = thread_offset
        .checked_add(offset as u64)
        .ok_or("Total offset overflow")?;
    
    // Convert to scalar
    let offset_bytes = total_offset.to_be_bytes();
    let mut scalar_bytes = [0u8; 32];
    scalar_bytes[24..32].copy_from_slice(&offset_bytes);
    
    let offset_scalar = Scalar::from_be_bytes(scalar_bytes)
        .map_err(|_| "Invalid scalar")?;
    
    // Add offset to base key
    base_key.add_tweak(&offset_scalar)
        .map_err(|e| format!("Key addition failed: {}", e))
}
