//! Models used in storage module.

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/paradigmxyz/reth/main/assets/reth-docs.png",
    html_favicon_url = "https://avatars0.githubusercontent.com/u/97369466?s=256",
    issue_tracker_base_url = "https://github.com/paradigmxyz/reth/issues/"
)]
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

/// Error returned by a changeset [`Decompress`](reth_codecs::Decompress) implementation when the
/// raw value buffer is shorter than the structurally-minimum encoding for that row.
///
/// The changeset `Compact` encoders (`AccountBeforeTx`/`StorageBeforeTx`) always emit at least the
/// fixed-size subkey prefix (an address, and for storage also the 32-byte key), so a buffer below
/// that minimum is never a legitimate row. It is produced by a torn read of an mmap'd static-file
/// changeset segment while the persistence thread is mid-append (a half-published offset table can
/// yield an in-bounds empty range). Surfacing it as an error — instead of panicking inside the
/// infallible `Compact::from_compact` slicing, or silently skipping the row and corrupting the
/// derived state root — lets the caller retry the read.
#[cfg(any(test, feature = "reth-codec"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncatedChangesetRow {
    /// The static-file segment the row belongs to (e.g. `"StorageChangeSets"`).
    pub segment: &'static str,
    /// The length of the observed (too-short) buffer.
    pub len: usize,
    /// The structural minimum length for a legitimate row of this segment.
    pub min: usize,
}

#[cfg(any(test, feature = "reth-codec"))]
impl core::fmt::Display for TruncatedChangesetRow {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "truncated {} changeset row: buffer length {} is below the structural minimum {} \
             (likely a torn read of a static file being concurrently appended)",
            self.segment, self.len, self.min
        )
    }
}

#[cfg(any(test, feature = "reth-codec"))]
impl core::error::Error for TruncatedChangesetRow {}

/// Accounts
pub mod accounts;
pub use accounts::AccountBeforeTx;

/// Blocks
pub mod blocks;
pub use blocks::{StaticFileBlockWithdrawals, StoredBlockBodyIndices, StoredBlockWithdrawals};

/// Storage
pub mod storage;
pub use storage::StorageBeforeTx;

/// Client Version
pub mod client_version;
pub use client_version::ClientVersion;
