use alloy_primitives::{Address, B256, U256};
use reth_primitives_traits::ValueWithSubKey;

/// Storage entry as it is saved in the static files.
///
/// [`B256`] is the subkey.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(any(test, feature = "arbitrary"), derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(any(test, feature = "reth-codec"), reth_codecs::add_arbitrary_tests(compact))]
pub struct StorageBeforeTx {
    /// Address for the storage entry. Acts as `DupSort::SubKey` in static files.
    pub address: Address,
    /// Storage key.
    pub key: B256,
    /// Value on storage key.
    pub value: U256,
}

impl ValueWithSubKey for StorageBeforeTx {
    type SubKey = B256;

    fn get_subkey(&self) -> Self::SubKey {
        self.key
    }
}

// NOTE: Removing reth_codec and manually encode subkey
// and compress second part of the value. If we have compression
// over whole value (Even SubKey) that would mess up fetching of values with seek_by_key_subkey
#[cfg(any(test, feature = "reth-codec"))]
impl reth_codecs::Compact for StorageBeforeTx {
    fn to_compact<B>(&self, buf: &mut B) -> usize
    where
        B: bytes::BufMut + AsMut<[u8]>,
    {
        buf.put_slice(self.address.as_slice());
        buf.put_slice(&self.key[..]);
        self.value.to_compact(buf) + 52
    }

    fn from_compact(buf: &[u8], len: usize) -> (Self, &[u8]) {
        let address = Address::from_slice(&buf[..20]);
        let key = B256::from_slice(&buf[20..52]);
        let (value, out) = U256::from_compact(&buf[52..], len - 52);
        (Self { address, key, value }, out)
    }
}

/// Structural minimum length of a `StorageBeforeTx` compact encoding: 20-byte address + 32-byte
/// key + at least 0 bytes for the `U256` value. See [`StorageBeforeTx::to_compact`].
#[cfg(any(test, feature = "reth-codec"))]
const STORAGE_BEFORE_TX_MIN_LEN: usize = 52;

// Hand-written `Compress`/`Decompress` (instead of `impl_compression_for_compact!`) so the decode
// boundary can guard against a torn read handing an empty/short buffer to the infallible
// `from_compact` (which would panic on `buf[..20]`/`buf[20..52]`).
#[cfg(any(test, feature = "reth-codec"))]
impl reth_codecs::Compress for StorageBeforeTx {
    type Compressed = alloc::vec::Vec<u8>;

    fn compress_to_buf<B: bytes::BufMut + AsMut<[u8]>>(&self, buf: &mut B) {
        let _ = reth_codecs::Compact::to_compact(self, buf);
    }
}

#[cfg(any(test, feature = "reth-codec"))]
impl reth_codecs::Decompress for StorageBeforeTx {
    fn decompress(value: &[u8]) -> Result<Self, reth_codecs::DecompressError> {
        if value.len() < STORAGE_BEFORE_TX_MIN_LEN {
            return Err(reth_codecs::DecompressError::new(crate::TruncatedChangesetRow {
                segment: "StorageChangeSets",
                len: value.len(),
                min: STORAGE_BEFORE_TX_MIN_LEN,
            }));
        }
        let (obj, _) = reth_codecs::Compact::from_compact(value, value.len());
        Ok(obj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reth_codecs::{Compress, Decompress};

    /// A well-formed minimal encoding (value `U256::ZERO`) is exactly the structural minimum and
    /// round-trips cleanly.
    #[test]
    fn decompress_minimal_row_ok() {
        let row = StorageBeforeTx { address: Address::ZERO, key: B256::ZERO, value: U256::ZERO };
        let bytes = row.compress();
        assert_eq!(bytes.len(), STORAGE_BEFORE_TX_MIN_LEN);
        assert_eq!(StorageBeforeTx::decompress(&bytes).unwrap(), row);
    }

    /// A non-trivial value still round-trips.
    #[test]
    fn decompress_roundtrip_nonzero() {
        let row = StorageBeforeTx {
            address: Address::repeat_byte(0xab),
            key: B256::repeat_byte(0xcd),
            value: U256::from(0x1234_5678u64),
        };
        let bytes = row.compress();
        assert!(bytes.len() >= STORAGE_BEFORE_TX_MIN_LEN);
        assert_eq!(StorageBeforeTx::decompress(&bytes).unwrap(), row);
    }

    /// An empty buffer (torn read of a segment being appended) surfaces a typed error, not a panic.
    #[test]
    fn decompress_empty_errors() {
        assert!(StorageBeforeTx::decompress(&[]).is_err());
    }

    /// Short-but-nonempty buffers below the 52-byte minimum also surface a typed error.
    #[test]
    fn decompress_short_errors() {
        for len in [1usize, 20, 51] {
            let buf = vec![0u8; len];
            assert!(
                StorageBeforeTx::decompress(&buf).is_err(),
                "expected error for buffer of length {len}"
            );
        }
    }
}
