use secp256k1::{SecretKey, PublicKey, SECP256K1};
use tiny_keccak::{Hasher, Keccak};

fn public_key_to_eth_address(public_key: &PublicKey) -> String {
    let public_key_bytes = public_key.serialize_uncompressed();
    let mut keccak = Keccak::v256();
    keccak.update(&public_key_bytes[1..]);
    let mut hash = [0u8; 32];
    keccak.finalize(&mut hash);
    let mut address_bytes = [0u8; 20];
    address_bytes.copy_from_slice(&hash[12..32]);
    format!("0x{}", hex::encode(address_bytes))
}

fn verify_keypair(private_key_hex: &str, expected_address: &str) -> bool {
    let private_bytes = hex::decode(private_key_hex).unwrap();
    let secret_key = SecretKey::from_slice(&private_bytes).unwrap();
    let public_key = PublicKey::from_secret_key(&SECP256K1, &secret_key);
    let generated_address = public_key_to_eth_address(&public_key);

    println!("Private Key: {}", private_key_hex);
    println!("Expected:    {}", expected_address);
    println!("Generated:   {}", generated_address);

    let matches = generated_address.to_lowercase() == expected_address.to_lowercase();
    println!("Match: {}\n", if matches { "✓ YES" } else { "✗ NO" });

    matches
}

fn main() {
    println!("=== Verifying Address Correctness ===\n");

    println!("Test 1: Prefix 'dead' address");
    let test1 = verify_keypair(
        "be2b0810f84752cf0c7115230fab3558c92c04d5cd731ffb3d728b523f06bcda",
        "0xdead84dd0b296e24722bd9a58fd112338fe8e798"
    );

    println!("Test 2: Suffix 'beef' address");
    let test2 = verify_keypair(
        "3827ccef0d34b985bf8856dd24afa7642fb2de2b409ec56d1f21e7c374751f99",
        "0x4e51ab1bd2bc493ed6be767c7915df7832a0beef"
    );

    if test1 && test2 {
        println!("✓ All addresses verified correctly!");
    } else {
        println!("✗ Address verification failed!");
        std::process::exit(1);
    }
}
