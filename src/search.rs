use crate::address::{SequentialGenerator, public_key_to_eth_address, private_key_to_hex};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

pub struct VanityConfig {
    pub prefix: Option<String>,
    pub suffix: Option<String>,
    pub threads: usize,
}

pub struct VanityResult {
    pub address: String,
    pub private_key: String,
}

pub struct FoundAddress {
    pub address: String,
    pub private_key: String,
}

/// CPU-based vanity address search
pub fn search_continuous(
    config: &VanityConfig,
    attempts: Arc<AtomicU64>,
    found: Arc<AtomicBool>,
) -> Option<FoundAddress> {
    let mut gen = SequentialGenerator::new();
    
    loop {
        if found.load(Ordering::Relaxed) {
            return None;
        }
        
        let (secret, public) = gen.next();
        let address = public_key_to_eth_address(public);
        
        attempts.fetch_add(1, Ordering::Relaxed);
        
        // Check pattern match
        let matches = if let Some(ref prefix) = config.prefix {
            // Skip "0x" prefix when checking
            address[2..].to_lowercase().starts_with(&prefix.to_lowercase())
        } else if let Some(ref suffix) = config.suffix {
            address.to_lowercase().ends_with(&suffix.to_lowercase())
        } else {
            false
        };
        
        if matches {
            if !found.swap(true, Ordering::SeqCst) {
                return Some(FoundAddress {
                    address: address.clone(),
                    private_key: private_key_to_hex(secret),
                });
            }
            return None;
        }
    }
}

/// Search with config
pub fn search_with_config(config: VanityConfig) -> Option<VanityResult> {
    let attempts = Arc::new(AtomicU64::new(0));
    let found = Arc::new(AtomicBool::new(false));
    
    let handles: Vec<_> = (0..config.threads)
        .map(|_| {
            let config = VanityConfig {
                prefix: config.prefix.clone(),
                suffix: config.suffix.clone(),
                threads: 1,
            };
            let attempts = Arc::clone(&attempts);
            let found = Arc::clone(&found);
            
            std::thread::spawn(move || {
                search_continuous(&config, attempts, found)
            })
        })
        .collect();
    
    for handle in handles {
        if let Ok(Some(result)) = handle.join() {
            return Some(VanityResult {
                address: result.address,
                private_key: result.private_key,
            });
        }
    }
    
    None
}

/// Parallel search
pub fn search_parallel(
    prefix: Option<String>,
    suffix: Option<String>,
    threads: usize,
) -> Option<VanityResult> {
    search_with_config(VanityConfig {
        prefix,
        suffix,
        threads,
    })
}

pub struct VanitySearcher;

impl VanitySearcher {
    pub fn search(config: VanityConfig) -> Option<VanityResult> {
        search_with_config(config)
    }
}
