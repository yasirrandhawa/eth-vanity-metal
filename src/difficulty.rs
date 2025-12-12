use std::time::Duration;

/// Calculate difficulty based on prefix/suffix patterns
/// Hex has 16 characters (0-9, a-f) case-insensitive
pub fn calculate_difficulty(prefix: Option<&str>, suffix: Option<&str>) -> u64 {
    let base: u64 = 16; // Hex characters

    let prefix_diff = prefix.map(|p| base.saturating_pow(p.len() as u32)).unwrap_or(1);
    let suffix_diff = suffix.map(|s| base.saturating_pow(s.len() as u32)).unwrap_or(1);

    prefix_diff.saturating_mul(suffix_diff)
}

/// Estimate time based on difficulty and speed
pub fn estimate_time(difficulty: u64, keys_per_sec: u64) -> Duration {
    if keys_per_sec == 0 {
        return Duration::from_secs(u64::MAX);
    }
    Duration::from_secs(difficulty / keys_per_sec)
}

/// Format difficulty number as "38.07B", "195.11K", etc.
pub fn format_difficulty(n: u64) -> String {
    if n >= 1_000_000_000_000 {
        format!("{:.2}T", n as f64 / 1_000_000_000_000.0)
    } else if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Format duration as "5.3 hours", "2.5 minutes", "45 seconds", etc.
pub fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();

    if secs >= 86400 {
        format!("{:.1} days", secs as f64 / 86400.0)
    } else if secs >= 3600 {
        format!("{:.1} hours", secs as f64 / 3600.0)
    } else if secs >= 60 {
        format!("{:.1} minutes", secs as f64 / 60.0)
    } else {
        format!("{} seconds", secs)
    }
}
