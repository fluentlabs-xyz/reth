use super::mask::{ColumnSelectorOne, ColumnSelectorThree, ColumnSelectorTwo};
use alloy_primitives::B256;
use derive_more::{Deref, DerefMut};
use reth_db_api::table::Decompress;
use reth_nippy_jar::{DataReader, NippyJar, NippyJarCursor};
use reth_static_file_types::SegmentHeader;
use reth_storage_errors::provider::{ProviderError, ProviderResult};
use std::sync::Arc;

/// Cursor of a static file segment.
#[derive(Debug, Deref, DerefMut)]
pub struct StaticFileCursor<'a>(NippyJarCursor<'a, SegmentHeader>);

/// Type alias for column results with optional values.
type ColumnResult<T> = ProviderResult<Option<T>>;

impl<'a> StaticFileCursor<'a> {
    /// Returns a new [`StaticFileCursor`].
    pub fn new(jar: &'a NippyJar<SegmentHeader>, reader: Arc<DataReader>) -> ProviderResult<Self> {
        Ok(Self(NippyJarCursor::with_reader(jar, reader).map_err(ProviderError::other)?))
    }

    /// Returns the current `BlockNumber` or `TxNumber` of the cursor depending on the kind of
    /// static file segment.
    pub fn number(&self) -> Option<u64> {
        self.jar().user_header().start().map(|start| self.row_index() + start)
    }

    /// Gets a row of values.
    pub fn get(
        &mut self,
        key_or_num: KeyOrNumber<'_>,
        mask: usize,
    ) -> ProviderResult<Option<Vec<&'_ [u8]>>> {
        if self.jar().rows() == 0 {
            return Ok(None)
        }

        let row = match key_or_num {
            KeyOrNumber::Key(_) => unimplemented!(),
            KeyOrNumber::Number(n) => match self.jar().user_header().start() {
                Some(offset) => {
                    if offset > n {
                        return Ok(None)
                    }
                    // Propagate jar-cursor errors (e.g. `NippyJarError::InconsistentData` from a
                    // torn read of a segment being concurrently appended) instead of collapsing
                    // them into `Ok(None)`: a silently skipped row would corrupt derived state,
                    // while an error lets the caller retry. A legitimately absent row (row index
                    // past the committed row count) is still `Ok(None)`.
                    self.row_by_number_with_cols((n - offset) as usize, mask)
                        .map_err(ProviderError::other)?
                }
                None => None,
            },
        };

        Ok(row)
    }

    /// Gets one column value from a row.
    pub fn get_one<M: ColumnSelectorOne>(
        &mut self,
        key_or_num: KeyOrNumber<'_>,
    ) -> ColumnResult<M::FIRST> {
        let row = self.get(key_or_num, M::MASK)?;

        match row {
            Some(row) => Ok(Some(M::FIRST::decompress(row[0])?)),
            None => Ok(None),
        }
    }

    /// Gets two column values from a row.
    pub fn get_two<M: ColumnSelectorTwo>(
        &mut self,
        key_or_num: KeyOrNumber<'_>,
    ) -> ColumnResult<(M::FIRST, M::SECOND)> {
        let row = self.get(key_or_num, M::MASK)?;

        match row {
            Some(row) => Ok(Some((M::FIRST::decompress(row[0])?, M::SECOND::decompress(row[1])?))),
            None => Ok(None),
        }
    }

    /// Gets three column values from a row.
    pub fn get_three<M: ColumnSelectorThree>(
        &mut self,
        key_or_num: KeyOrNumber<'_>,
    ) -> ColumnResult<(M::FIRST, M::SECOND, M::THIRD)> {
        let row = self.get(key_or_num, M::MASK)?;

        match row {
            Some(row) => Ok(Some((
                M::FIRST::decompress(row[0])?,
                M::SECOND::decompress(row[1])?,
                M::THIRD::decompress(row[2])?,
            ))),
            None => Ok(None),
        }
    }
}

/// Either a key _or_ a block/tx number
#[derive(Debug)]
pub enum KeyOrNumber<'a> {
    /// A slice used as a key. Usually a block/tx hash
    Key(&'a [u8]),
    /// A block/tx number
    Number(u64),
}

impl<'a> From<&'a B256> for KeyOrNumber<'a> {
    fn from(value: &'a B256) -> Self {
        KeyOrNumber::Key(value.as_slice())
    }
}

impl From<u64> for KeyOrNumber<'_> {
    fn from(value: u64) -> Self {
        KeyOrNumber::Number(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::static_file::AccountChangesetMask;
    use reth_db_api::models::AccountBeforeTx;
    use reth_nippy_jar::NippyJarWriter;
    use reth_static_file_types::{SegmentRangeInclusive, StaticFileSegment};

    /// A jar-cursor error (e.g. `NippyJarError::InconsistentData` from a torn read of a segment
    /// being concurrently appended) must propagate out of `get_one` as `Err`, not be swallowed
    /// into `Ok(None)`: a silently skipped changeset row would corrupt derived state roots.
    #[test]
    fn get_one_propagates_jar_cursor_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("static_file_account-change-sets_0_0");

        let header = SegmentHeader::new(
            SegmentRangeInclusive::new(0, 0),
            Some(SegmentRangeInclusive::new(0, 0)),
            None,
            StaticFileSegment::AccountChangeSets,
        );

        // Two single-column rows, each a valid minimal `AccountBeforeTx` (bare 20-byte address).
        let row = [0u8; 20];
        {
            let jar = NippyJar::new(1, &path, header);
            let mut writer = NippyJarWriter::new(jar).unwrap();
            writer.append_column(Some(Ok(&row[..]))).unwrap();
            writer.append_column(Some(Ok(&row[..]))).unwrap();
            writer.commit().unwrap();
        }

        // Sanity: a well-formed jar reads the row back.
        {
            let jar = NippyJar::<SegmentHeader>::load(&path).unwrap();
            let reader = Arc::new(jar.open_data_reader().unwrap());
            let mut cursor = StaticFileCursor::new(&jar, reader).unwrap();
            let value = cursor.get_one::<AccountChangesetMask>(0.into()).unwrap();
            assert_eq!(value, Some(AccountBeforeTx::default()));
        }

        // Truncate the data file below the committed offsets (the on-disk analog of a torn
        // offsets-vs-data read). Reading row 0 now yields an out-of-bounds data range.
        std::fs::OpenOptions::new().write(true).open(&path).unwrap().set_len(1).unwrap();

        let jar = NippyJar::<SegmentHeader>::load(&path).unwrap();
        let reader = Arc::new(jar.open_data_reader().unwrap());
        let mut cursor = StaticFileCursor::new(&jar, reader).unwrap();

        let result = cursor.get_one::<AccountChangesetMask>(0.into());
        assert!(
            matches!(result, Err(ProviderError::Other(_))),
            "expected the jar-cursor error to propagate, got {result:?}"
        );
    }
}
