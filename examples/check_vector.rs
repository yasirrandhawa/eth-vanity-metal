use secp256k1::{SecretKey, PublicKey, SECP256K1};
use tiny_keccak::{Hasher, Keccak};

fn public_key_to_eth_address(public_key: &PublicKey) -> String {
    let public_key_bytes = public_key.serialize_uncompressed();

    println!("Public key (uncompressed): {}", hex::encode(&public_key_bytes));
    println!("Public key length: {}", public_key_bytes.len());
    println!("Using bytes 1-65 for hashing");

    let mut keccak = Keccak::v256();
    keccak.update(&public_key_bytes[1..]); // Skip first byte (0x04)
    let mut hash = [0u8; 32];
    keccak.finalize(&mut hash);

    println!("Keccak hash: {}", hex::encode(&hash));

    let mut address_bytes = [0u8; 20];
    address_bytes.copy_from_slice(&hash[12..32]); // Take last 20 bytes

    format!("0x{}", hex::encode(address_bytes))
}

fn main() {
    let private_key_hex = "4c0883a69102937d6231471b5dbb6204fe512961708279f8b9f0629fbd2b6f72";

    println!("Testing known vector:");
    println!("Private key: {}", private_key_hex);
    println!();

    let private_bytes = hex::decode(private_key_hex).unwrap();
    let secret_key = SecretKey::from_slice(&private_bytes).unwrap();
    let public_key = PublicKey::from_secret_key(&SECP256K1, &secret_key);

    let address = public_key_to_eth_address(&public_key);

    println!("\nGenerated address: {}", address);
}
