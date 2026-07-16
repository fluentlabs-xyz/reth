use alloy_primitives::Address;
use reth_primitives_traits::{Account, ValueWithSubKey};

/// Account as it is saved in the database.
///
/// [`Address`] is the subkey.
#[derive(Debug, Default, Clone, Eq, PartialEq)]
#[cfg_attr(any(test, feature = "arbitrary"), derive(arbitrary::Arbitrary))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(any(test, feature = "reth-codec"), reth_codecs::add_arbitrary_tests(compact))]
pub struct AccountBeforeTx {
    /// Address for the account. Acts as `DupSort::SubKey`.
    pub address: Address,
    /// Account state before the transaction.
    pub info: Option<Account>,
}

impl ValueWithSubKey for AccountBeforeTx {
    type SubKey = Address;

    fn get_subkey(&self) -> Self::SubKey {
        self.address
    }
}

// NOTE: Removing reth_codec and manually encode subkey
// and compress second part of the value. If we have compression
// over whole value (Even SubKey) that would mess up fetching of values with seek_by_key_subkey
#[cfg(any(test, feature = "reth-codec"))]
impl reth_codecs::Compact for AccountBeforeTx {
    fn to_compact<B>(&self, buf: &mut B) -> usize
    where
        B: bytes::BufMut + AsMut<[u8]>,
    {
        // for now put full bytes and later compress it.
        buf.put_slice(self.address.as_slice());

        let acc_len = if let Some(account) = self.info { account.to_compact(buf) } else { 0 };
        acc_len + 20
    }

    fn from_compact(mut buf: &[u8], len: usize) -> (Self, &[u8]) {
        use bytes::Buf;
        let address = Address::from_slice(&buf[..20]);
        buf.advance(20);

        let info = (len - 20 > 0).then(|| {
            let (acc, advanced_buf) = Account::from_compact(buf, len - 20);
            buf = advanced_buf;
            acc
        });

        (Self { address, info }, buf)
    }
}

/// Structural minimum length of an `AccountBeforeTx` compact encoding: 20-byte address, with the
/// optional `Account` contributing 0 or more bytes. See [`AccountBeforeTx::to_compact`].
#[cfg(any(test, feature = "reth-codec"))]
const ACCOUNT_BEFORE_TX_MIN_LEN: usize = 20;

// Hand-written `Compress`/`Decompress` (instead of `impl_compression_for_compact!`) so the decode
// boundary can guard against a torn read handing an empty/short buffer to the infallible
// `from_compact` (which would panic on `buf[..20]` and underflow on `len - 20`).
#[cfg(any(test, feature = "reth-codec"))]
impl reth_codecs::Compress for AccountBeforeTx {
    type Compressed = alloc::vec::Vec<u8>;

    fn compress_to_buf<B: bytes::BufMut + AsMut<[u8]>>(&self, buf: &mut B) {
        let _ = reth_codecs::Compact::to_compact(self, buf);
    }
}

#[cfg(any(test, feature = "reth-codec"))]
impl reth_codecs::Decompress for AccountBeforeTx {
    fn decompress(value: &[u8]) -> Result<Self, reth_codecs::DecompressError> {
        if value.len() < ACCOUNT_BEFORE_TX_MIN_LEN {
            return Err(reth_codecs::DecompressError::new(crate::TruncatedChangesetRow {
                segment: "AccountChangeSets",
                len: value.len(),
                min: ACCOUNT_BEFORE_TX_MIN_LEN,
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

    /// A well-formed minimal encoding (`info: None`) is exactly the structural minimum (the bare
    /// 20-byte address) and round-trips cleanly.
    #[test]
    fn decompress_minimal_row_ok() {
        let row = AccountBeforeTx { address: Address::ZERO, info: None };
        let bytes = row.clone().compress();
        assert_eq!(bytes.len(), ACCOUNT_BEFORE_TX_MIN_LEN);
        assert_eq!(AccountBeforeTx::decompress(&bytes).unwrap(), row);
    }

    /// A row carrying account info still round-trips.
    #[test]
    fn decompress_roundtrip_with_info() {
        use reth_primitives_traits::Account;
        let row = AccountBeforeTx {
            address: Address::repeat_byte(0x11),
            info: Some(Account { nonce: 7, balance: alloy_primitives::U256::from(42u64), bytecode_hash: None }),
        };
        let bytes = row.clone().compress();
        assert!(bytes.len() >= ACCOUNT_BEFORE_TX_MIN_LEN);
        assert_eq!(AccountBeforeTx::decompress(&bytes).unwrap(), row);
    }

    /// An empty buffer (torn read of a segment being appended) surfaces a typed error, not a panic.
    #[test]
    fn decompress_empty_errors() {
        assert!(AccountBeforeTx::decompress(&[]).is_err());
    }

    /// Short-but-nonempty buffers below the 20-byte minimum also surface a typed error.
    #[test]
    fn decompress_short_errors() {
        for len in [1usize, 10, 19] {
            let buf = vec![0u8; len];
            assert!(
                AccountBeforeTx::decompress(&buf).is_err(),
                "expected error for buffer of length {len}"
            );
        }
    }
}
