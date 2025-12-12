/// GPU hashing pipeline for batch Keccak-256 operations
///
/// This module provides the GpuHasher which manages Metal buffers and compute pipelines
/// for accelerated Keccak-256 hashing of public keys.

use super::{MetalContext, GpuError};
use metal::{ComputePipelineState, MTLResourceOptions, MTLSize};
use std::sync::Arc;

/// GPU hasher for batch Keccak-256 operations
pub struct GpuHasher {
    context: Arc<MetalContext>,
    pipeline_state: ComputePipelineState,
    max_batch_size: usize,
}

impl GpuHasher {
    /// Create a new GPU hasher with the given context
    pub fn new(_context: Arc<MetalContext>) -> Result<Self, GpuError> {
        // Hybrid GPU mode not implemented for ETH
        // Use native_search.rs (--gpu-native flag) for full GPU acceleration
        Err(GpuError::InitializationFailed(
            "Hybrid GPU mode not implemented. Use --gpu-native instead".to_string()
        ))
    }

    /// Hash a batch of 64-byte public key inputs
    ///
    /// # Arguments
    /// * `inputs` - Vector of 64-byte public key data (without 0x04 prefix)
    ///
    /// # Returns
    /// Vector of 32-byte Keccak-256 hashes
    pub fn hash_batch(&self, inputs: &[[u8; 64]]) -> Result<Vec<[u8; 32]>, GpuError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = inputs.len().min(self.max_batch_size);

        // Create input buffer (64 bytes per input)
        let input_size = batch_size * 64;
        let input_buffer = self.context.device.new_buffer_with_bytes_no_copy(
            inputs.as_ptr() as *const _,
            input_size as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        // Create output buffer (32 bytes per output)
        let output_size = batch_size * 32;
        let output_buffer = self.context.device.new_buffer(
            output_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create batch size buffer
        let batch_size_u32 = batch_size as u32;
        let batch_size_buffer = self.context.device.new_buffer_with_bytes_no_copy(
            &batch_size_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        // Create command buffer
        let command_buffer = self.context.command_queue.new_command_buffer();

        // Create compute encoder
        let compute_encoder = command_buffer.new_compute_command_encoder();

        // Set pipeline state
        compute_encoder.set_compute_pipeline_state(&self.pipeline_state);

        // Set buffers
        compute_encoder.set_buffer(0, Some(&input_buffer), 0);
        compute_encoder.set_buffer(1, Some(&output_buffer), 0);
        compute_encoder.set_buffer(2, Some(&batch_size_buffer), 0);

        // Calculate thread groups
        let thread_execution_width = self.pipeline_state.thread_execution_width();
        let max_total_threads = self.pipeline_state.max_total_threads_per_threadgroup();

        // Use optimal threadgroup size
        let threadgroup_size = thread_execution_width.min(max_total_threads).min(256) as usize;

        let threadgroups_count = (batch_size + threadgroup_size - 1) / threadgroup_size;

        let threadgroup_size_metal = MTLSize {
            width: threadgroup_size as u64,
            height: 1,
            depth: 1,
        };

        let threadgroups_per_grid = MTLSize {
            width: threadgroups_count as u64,
            height: 1,
            depth: 1,
        };

        // Dispatch compute kernel
        compute_encoder.dispatch_thread_groups(threadgroups_per_grid, threadgroup_size_metal);

        // End encoding
        compute_encoder.end_encoding();

        // Commit and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results from output buffer
        let output_ptr = output_buffer.contents() as *const u8;
        let mut results = Vec::with_capacity(batch_size);

        unsafe {
            for i in 0..batch_size {
                let mut hash = [0u8; 32];
                std::ptr::copy_nonoverlapping(
                    output_ptr.add(i * 32),
                    hash.as_mut_ptr(),
                    32,
                );
                results.push(hash);
            }
        }

        Ok(results)
    }

    /// Get maximum supported batch size
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Get GPU device name
    pub fn device_name(&self) -> String {
        self.context.device_name()
    }
}

/// Create a GPU hasher instance
pub fn create_hasher(context: Arc<MetalContext>) -> Result<GpuHasher, GpuError> {
    GpuHasher::new(context)
}
