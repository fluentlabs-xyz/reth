//! Trie changeset computation and caching utilities.
//!
//! This module provides functionality to compute trie changesets for a given block,
//! which represent the old trie node values before the block was processed.
//!
//! It also provides an efficient in-memory cache for these changesets, which is essential for:
//! - **Reorg support**: Quickly access changesets to revert blocks during chain reorganizations
//! - **Memory efficiency**: Automatic eviction ensures bounded memory usage

use crate::{
    DatabaseHashedCursorFactory, DatabaseStateRoot, DatabaseTrieCursorFactory, TrieTableAdapter,
};
use alloy_primitives::{map::B256Map, BlockNumber, B256};
use parking_lot::RwLock;
use reth_primitives_traits::FastInstant as Instant;
use reth_storage_api::{
    BlockNumReader, ChangeSetReader, DBProvider, StageCheckpointReader, StorageChangeSetReader,
    StorageSettingsCache,
};
use reth_storage_errors::provider::{ProviderError, ProviderResult};
use reth_trie::{
    changesets::compute_trie_changesets,
    trie_cursor::{InMemoryTrieCursorFactory, TrieCursor, TrieCursorFactory},
    HashedPostStateSorted, TrieInputSorted,
};
use reth_trie_common::updates::{StorageTrieUpdatesSorted, TrieUpdatesSorted};
use std::{
    collections::BTreeMap,
    fmt,
    ops::RangeInclusive,
    sync::{Arc, OnceLock},
};
use tracing::{debug, debug_span, warn};

#[cfg(feature = "metrics")]
use reth_metrics::{
    metrics::{Counter, Gauge},
    Metrics,
};

/// Computes trie changesets for a block.
///
/// # Algorithm
///
/// For block N:
/// 1. Query cumulative `HashedPostState` revert for block N-1 (from db tip to after N-1)
/// 2. Use that to calculate cumulative `TrieUpdates` revert for block N-1
/// 3. Query per-block `HashedPostState` revert for block N
/// 4. Create prefix sets from the per-block revert (step 3)
/// 5. Create overlay with cumulative trie updates and cumulative state revert for N-1
/// 6. Calculate trie updates for block N using the overlay and per-block `HashedPostState`.
/// 7. Compute changesets using the N-1 overlay and the newly calculated trie updates for N
///
/// # Arguments
///
/// * `provider` - Database provider with changeset access
/// * `block_number` - Block number to compute changesets for
///
/// # Returns
///
/// Changesets (old trie node values) for the specified block
///
/// # Errors
///
/// Returns error if:
/// - Block number exceeds database tip (based on Finish stage checkpoint)
/// - Database access fails
/// - State root computation fails
pub fn compute_block_trie_changesets<Provider>(
    provider: &Provider,
    block_number: BlockNumber,
) -> Result<TrieUpdatesSorted, ProviderError>
where
    Provider: DBProvider
        + StageCheckpointReader
        + ChangeSetReader
        + StorageChangeSetReader
        + BlockNumReader
        + StorageSettingsCache,
{
    // Single-block entry point: read both reverts from the database (the cumulative revert
    // spans `(block+1)..=db_tip` and is read from scratch here). The range walk in
    // `get_or_compute_range` instead maintains the cumulative revert incrementally and calls
    // [`compute_block_trie_changesets_from_reverts`] directly to avoid the per-block
    // from-scratch suffix read.
    let individual_state_revert =
        crate::state::from_reverts_auto(provider, block_number..=block_number)?;
    let cumulative_state_revert = crate::state::from_reverts_auto(provider, (block_number + 1)..)?;

    compute_block_trie_changesets_from_reverts(
        provider,
        block_number,
        &individual_state_revert,
        &cumulative_state_revert,
    )
}

/// Computes trie changesets for a block from pre-collected state reverts.
///
/// This is the shared core of [`compute_block_trie_changesets`]. It takes the state reverts as
/// arguments instead of reading them from the database, which lets the range walk in
/// [`ChangesetCache::get_or_compute_range`] supply an *incrementally accumulated* cumulative
/// revert rather than recomputing the whole `(block+1)..=db_tip` suffix from scratch for every
/// block (an O(range²) walk → O(range)).
///
/// # Arguments
///
/// * `individual_state_revert` — the per-block revert, i.e. `from_reverts(block..=block)`.
/// * `cumulative_state_revert` — the cumulative revert `from_reverts((block+1)..=db_tip)`,
///   representing the state as it was just after `block` was processed.
///
/// The two reverts are mathematically identical whether read from scratch or accumulated
/// incrementally (see the differential test), because `extend_ref_and_sort` gives the older
/// block precedence — the same precedence the cumulative-suffix read uses (oldest occurrence
/// wins).
pub(crate) fn compute_block_trie_changesets_from_reverts<Provider>(
    provider: &Provider,
    block_number: BlockNumber,
    individual_state_revert: &HashedPostStateSorted,
    cumulative_state_revert: &HashedPostStateSorted,
) -> Result<TrieUpdatesSorted, ProviderError>
where
    Provider: DBProvider
        + StageCheckpointReader
        + ChangeSetReader
        + StorageChangeSetReader
        + BlockNumReader
        + StorageSettingsCache,
{
    crate::with_adapter!(provider, |A| {
        compute_block_trie_changesets_from_reverts_inner::<_, A>(
            provider,
            block_number,
            individual_state_revert,
            cumulative_state_revert,
        )
    })
}

fn compute_block_trie_changesets_from_reverts_inner<Provider, A>(
    provider: &Provider,
    block_number: BlockNumber,
    individual_state_revert: &HashedPostStateSorted,
    cumulative_state_revert: &HashedPostStateSorted,
) -> Result<TrieUpdatesSorted, ProviderError>
where
    Provider: DBProvider
        + StageCheckpointReader
        + ChangeSetReader
        + StorageChangeSetReader
        + BlockNumReader
        + StorageSettingsCache,
    A: TrieTableAdapter,
{
    debug!(
        target: "trie::changeset_cache",
        block_number,
        "Computing block trie changesets from database state"
    );

    // Step 1: Collect/calculate state reverts (supplied by the caller).
    // `cumulative_state_revert` reverts all changes from db tip back to just after `block`
    // was processed; `individual_state_revert` is only the changes from this block.

    // This reverts all changes from db tip back to just after block-1 was processed
    let mut cumulative_state_revert_prev = cumulative_state_revert.clone();
    cumulative_state_revert_prev.extend_ref_and_sort(individual_state_revert);

    // Step 2: Calculate cumulative trie updates revert for block-1
    // This gives us the trie state as it was after block-1 was processed
    let prefix_sets_prev = cumulative_state_revert_prev.construct_prefix_sets();
    let input_prev = TrieInputSorted::new(
        Arc::default(),
        Arc::new(cumulative_state_revert_prev),
        prefix_sets_prev,
    );

    type DbStateRoot<'a, TX, A> = reth_trie::StateRoot<
        DatabaseTrieCursorFactory<&'a TX, A>,
        DatabaseHashedCursorFactory<&'a TX>,
    >;

    let cumulative_trie_updates_prev =
        DbStateRoot::<_, A>::overlay_root_from_nodes_with_updates(provider.tx_ref(), input_prev)
            .map_err(ProviderError::other)?
            .1
            .into_sorted();

    // Step 3: Create prefix sets from individual revert (only paths changed by this block)
    let prefix_sets = individual_state_revert.construct_prefix_sets();

    // Step 4: Calculate trie updates for block
    // Use cumulative trie updates for block-1 as the node overlay and cumulative state for block
    let input = TrieInputSorted::new(
        Arc::new(cumulative_trie_updates_prev.clone()),
        Arc::new(cumulative_state_revert.clone()),
        prefix_sets,
    );

    let trie_updates =
        DbStateRoot::<_, A>::overlay_root_from_nodes_with_updates(provider.tx_ref(), input)
            .map_err(ProviderError::other)?
            .1
            .into_sorted();

    // Step 5: Compute changesets using cumulative trie updates for block-1 as overlay
    // Create an overlay cursor factory that has the trie state from after block-1
    let db_cursor_factory = DatabaseTrieCursorFactory::<_, A>::new(provider.tx_ref());
    let overlay_factory =
        InMemoryTrieCursorFactory::new(db_cursor_factory, &cumulative_trie_updates_prev);

    let changesets =
        compute_trie_changesets(&overlay_factory, &trie_updates).map_err(ProviderError::other)?;

    debug!(
        target: "trie::changeset_cache",
        block_number,
        num_account_nodes = changesets.account_nodes_ref().len(),
        num_storage_tries = changesets.storage_tries_ref().len(),
        "Computed block trie changesets successfully"
    );

    Ok(changesets)
}

/// Computes block trie updates using the changeset cache.
///
/// # Algorithm
///
/// For block N:
/// 1. Get cumulative trie reverts from block N+1 to db tip using the cache
/// 2. Create an overlay cursor factory with these reverts (representing trie state after block N)
/// 3. Walk through account trie changesets for block N
/// 4. For each changed path, look up the current value using the overlay cursor
/// 5. Walk through storage trie changesets for block N
/// 6. For each changed path, look up the current value using the overlay cursor
/// 7. Return the collected trie updates
///
/// # Arguments
///
/// * `cache` - Handle to the changeset cache for retrieving trie reverts
/// * `provider` - Database provider for accessing changesets and block data
/// * `block_number` - Block number to compute trie updates for
///
/// # Returns
///
/// Trie updates representing the state of trie nodes after the block was processed
///
/// # Errors
///
/// Returns error if:
/// - Block number exceeds database tip
/// - Database access fails
/// - Cache retrieval fails
pub fn compute_block_trie_updates<Provider>(
    cache: &ChangesetCache,
    provider: &Provider,
    block_number: BlockNumber,
) -> ProviderResult<TrieUpdatesSorted>
where
    Provider: DBProvider
        + StageCheckpointReader
        + ChangeSetReader
        + StorageChangeSetReader
        + BlockNumReader
        + StorageSettingsCache,
{
    crate::with_adapter!(provider, |A| {
        compute_block_trie_updates_inner::<_, A>(cache, provider, block_number)
    })
}

fn compute_block_trie_updates_inner<Provider, A>(
    cache: &ChangesetCache,
    provider: &Provider,
    block_number: BlockNumber,
) -> ProviderResult<TrieUpdatesSorted>
where
    Provider: DBProvider
        + StageCheckpointReader
        + ChangeSetReader
        + StorageChangeSetReader
        + BlockNumReader
        + StorageSettingsCache,
    A: TrieTableAdapter,
{
    let tx = provider.tx_ref();

    // Get the database tip block number
    let db_tip_block = provider
        .get_stage_checkpoint(reth_stages_types::StageId::Finish)?
        .as_ref()
        .map(|chk| chk.block_number)
        .ok_or_else(|| ProviderError::InsufficientChangesets {
            requested: block_number,
            available: 0..=0,
        })?;

    // Step 1: Get the block hash for the target block
    let block_hash = provider.block_hash(block_number)?.ok_or_else(|| {
        ProviderError::other(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("block hash not found for block number {}", block_number),
        ))
    })?;

    // Step 2: Get the trie changesets for the target block from cache
    let changesets = cache.get_or_compute(block_hash, block_number, provider)?;

    // Step 3: Get the trie reverts for the state after the target block using the cache
    let reverts = cache.get_or_compute_range(provider, (block_number + 1)..=db_tip_block)?;

    // Step 4: Create an InMemoryTrieCursorFactory with the reverts
    // This gives us the trie state as it was after the target block was processed
    let db_cursor_factory = DatabaseTrieCursorFactory::<_, A>::new(tx);
    let cursor_factory = InMemoryTrieCursorFactory::new(db_cursor_factory, &reverts);

    // Step 5: Collect all account trie nodes that changed in the target block
    let account_nodes_ref = changesets.account_nodes_ref();
    let mut account_nodes = Vec::with_capacity(account_nodes_ref.len());
    let mut account_cursor = cursor_factory.account_trie_cursor()?;

    // Iterate over the account nodes from the changesets
    for (nibbles, _old_node) in account_nodes_ref {
        // Look up the current value of this trie node using the overlay cursor
        let node_value = account_cursor.seek_exact(*nibbles)?.map(|(_, node)| node);
        account_nodes.push((*nibbles, node_value));
    }

    // Step 6: Collect all storage trie nodes that changed in the target block
    let mut storage_tries = B256Map::default();

    // Iterate over the storage tries from the changesets
    for (hashed_address, storage_changeset) in changesets.storage_tries_ref() {
        let mut storage_cursor = cursor_factory.storage_trie_cursor(*hashed_address)?;
        let storage_nodes_ref = storage_changeset.storage_nodes_ref();
        let mut storage_nodes = Vec::with_capacity(storage_nodes_ref.len());

        // Iterate over the storage nodes for this account
        for (nibbles, _old_node) in storage_nodes_ref {
            // Look up the current value of this storage trie node
            let node_value = storage_cursor.seek_exact(*nibbles)?.map(|(_, node)| node);
            storage_nodes.push((*nibbles, node_value));
        }

        storage_tries.insert(
            *hashed_address,
            StorageTrieUpdatesSorted { storage_nodes, is_deleted: storage_changeset.is_deleted },
        );
    }

    Ok(TrieUpdatesSorted::new(account_nodes, storage_tries))
}

/// A pending changeset computation that other threads can wait on.
///
/// When a deferred trie task starts computing changesets for a block, it registers
/// a pending entry. If another thread needs the same changeset before the computation
/// finishes, it waits on this entry instead of falling back to the expensive
/// DB-based computation.
struct PendingChangeset {
    /// `None` when cancelled (e.g. due to panic), `Some(..)` when resolved with data.
    result: OnceLock<Option<Arc<TrieUpdatesSorted>>>,
}

impl PendingChangeset {
    const fn new() -> Self {
        Self { result: OnceLock::new() }
    }

    /// Blocks until the computation finishes. Returns `Some` if resolved with data,
    /// `None` if the computation was cancelled.
    fn wait(&self) -> Option<Arc<TrieUpdatesSorted>> {
        let _span =
            debug_span!(target: "trie::changeset_cache", "waiting_for_pending_changeset").entered();
        self.result.wait().clone()
    }

    /// Resolves the pending computation with the given result, waking all waiters.
    fn resolve(&self, changesets: Arc<TrieUpdatesSorted>) {
        let _ = self.result.set(Some(changesets));
    }

    /// Cancels the pending computation, waking all waiters so they fall through
    /// to the DB fallback.
    fn cancel(&self) {
        let _ = self.result.set(None);
    }
}

impl fmt::Debug for PendingChangeset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let is_resolved = self.result.get().is_some();
        f.debug_struct("PendingChangeset").field("resolved", &is_resolved).finish()
    }
}

/// Thread-safe changeset cache.
///
/// This type wraps a shared, mutable reference to the cache inner.
/// The `RwLock` enables concurrent reads while ensuring exclusive access for writes.
#[derive(Debug, Clone)]
pub struct ChangesetCache {
    inner: Arc<RwLock<ChangesetCacheInner>>,
}

impl Default for ChangesetCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ChangesetCache {
    /// Creates a new cache.
    ///
    /// The cache has no capacity limit and relies on explicit eviction
    /// via the `evict()` method to manage memory usage.
    pub fn new() -> Self {
        Self { inner: Arc::new(RwLock::new(ChangesetCacheInner::new())) }
    }

    /// Retrieves changesets for a block by hash.
    ///
    /// Returns `None` if the block is not in the cache (either evicted or never computed).
    /// Updates hit/miss metrics accordingly.
    pub fn get(&self, block_hash: &B256) -> Option<Arc<TrieUpdatesSorted>> {
        self.inner.read().get(block_hash)
    }

    /// Inserts changesets for a block into the cache.
    ///
    /// Also resolves any pending computation for this block hash, waking threads
    /// that are waiting for the result.
    ///
    /// This method does not perform any eviction. Eviction must be explicitly
    /// triggered by calling `evict()`.
    ///
    /// # Arguments
    ///
    /// * `block_hash` - Hash of the block
    /// * `block_number` - Block number for tracking and eviction
    /// * `changesets` - Trie changesets to cache
    fn insert(&self, block_hash: B256, block_number: u64, changesets: Arc<TrieUpdatesSorted>) {
        let pending = {
            let mut cache = self.inner.write();
            cache.insert(block_hash, block_number, Arc::clone(&changesets));
            cache.pending.remove(&block_hash)
        };

        // Resolve pending entry outside the write lock to avoid holding it
        // while waiters wake up.
        if let Some(pending) = pending {
            pending.resolve(changesets);
        }
    }

    /// Registers a pending changeset computation for the given block hash.
    ///
    /// Call this before starting changeset computation so that concurrent
    /// readers can wait for the result instead of falling back to the expensive
    /// DB-based computation.
    ///
    /// The returned [`PendingChangesetGuard`] must be used to resolve or cancel
    /// the pending entry. If dropped without resolving (e.g. due to a panic),
    /// the pending entry is automatically removed from the cache so that
    /// waiters fall through to the DB fallback.
    pub fn register_pending(&self, block_hash: B256) -> PendingChangesetGuard {
        let pending = Arc::new(PendingChangeset::new());
        let prev = self.inner.write().pending.insert(block_hash, Arc::clone(&pending));
        debug_assert!(prev.is_none(), "duplicate pending changeset for {block_hash:?}");
        PendingChangesetGuard { cache: self.clone(), block_hash, pending: Some(pending) }
    }

    /// Evicts changesets for blocks below the given block number.
    ///
    /// This should be called after blocks are persisted to the database to free
    /// memory for changesets that are no longer needed in the cache.
    ///
    /// # Arguments
    ///
    /// * `up_to_block` - Evict blocks with number < this value. Blocks with number >= this value
    ///   are retained.
    pub fn evict(&self, up_to_block: BlockNumber) {
        self.inner.write().evict(up_to_block)
    }

    /// Gets changesets from cache, or computes them on-the-fly if missing.
    ///
    /// This is the primary API for retrieving changesets. It checks three sources in order:
    /// 1. **Cache hit** — returns immediately
    /// 2. **Pending computation** — blocks until the deferred trie task finishes
    /// 3. **DB fallback** — computes from database state (expensive)
    ///
    /// # Arguments
    ///
    /// * `block_hash` - Hash of the block to get changesets for
    /// * `block_number` - Block number (for cache insertion and logging)
    /// * `provider` - Database provider for DB access
    ///
    /// # Returns
    ///
    /// Changesets for the block, either from cache, a pending computation, or computed on-the-fly
    pub fn get_or_compute<P>(
        &self,
        block_hash: B256,
        block_number: u64,
        provider: &P,
    ) -> ProviderResult<Arc<TrieUpdatesSorted>>
    where
        P: DBProvider
            + StageCheckpointReader
            + ChangeSetReader
            + StorageChangeSetReader
            + BlockNumReader
            + StorageSettingsCache,
    {
        self.get_or_compute_with(block_hash, block_number, || {
            compute_block_trie_changesets(provider, block_number)
        })
    }

    /// Cache/pending/compute core shared by [`Self::get_or_compute`] and the range walk.
    ///
    /// Checks the cache and any pending computation first; only on a genuine miss does it invoke
    /// `compute` (which produces the changesets from the database) and insert the result.
    ///
    /// The range walk uses this to supply an incrementally accumulated cumulative revert to the
    /// computation instead of the per-block from-scratch suffix read.
    fn get_or_compute_with(
        &self,
        block_hash: B256,
        block_number: u64,
        compute: impl FnOnce() -> Result<TrieUpdatesSorted, ProviderError>,
    ) -> ProviderResult<Arc<TrieUpdatesSorted>> {
        // Try cache first, and if missing, check for a pending computation.
        let pending = {
            let cache = self.inner.read();
            if let Some(changesets) = cache.get(&block_hash) {
                debug!(
                    target: "trie::changeset_cache",
                    ?block_hash,
                    block_number,
                    "Changeset cache HIT"
                );
                return Ok(changesets);
            }
            cache.pending.get(&block_hash).cloned()
        };

        // If there's a pending computation, wait for it instead of computing from DB.
        if let Some(pending) = pending {
            debug!(
                target: "trie::changeset_cache",
                ?block_hash,
                block_number,
                "Changeset cache MISS but pending computation found, waiting"
            );

            let start = Instant::now();

            if let Some(changesets) = pending.wait() {
                debug!(
                    target: "trie::changeset_cache",
                    ?block_hash,
                    block_number,
                    elapsed = ?start.elapsed(),
                    "Pending changeset resolved"
                );
                return Ok(changesets);
            }

            debug!(
                target: "trie::changeset_cache",
                ?block_hash,
                block_number,
                elapsed = ?start.elapsed(),
                "Pending changeset was cancelled, falling through to DB computation"
            );
        }

        // No cache hit and no pending computation - compute from database
        warn!(
            target: "trie::changeset_cache",
            ?block_hash,
            block_number,
            "Changeset cache MISS, falling back to DB-based computation"
        );

        let start = Instant::now();

        // Compute changesets
        let changesets = compute()?;

        let changesets = Arc::new(changesets);
        let elapsed = start.elapsed();

        debug!(
            target: "trie::changeset_cache",
            ?elapsed,
            block_number,
            ?block_hash,
            "Changeset computed from database and inserting into cache"
        );

        // Store in cache (with write lock)
        self.insert(block_hash, block_number, Arc::clone(&changesets));

        debug!(
            target: "trie::changeset_cache",
            ?block_hash,
            block_number,
            "Changeset successfully cached"
        );

        Ok(changesets)
    }

    /// Gets or computes accumulated trie reverts for a range of blocks.
    ///
    /// This method retrieves and accumulates all trie changesets (reverts) for the specified
    /// block range (inclusive). The changesets are accumulated in reverse order (newest to oldest)
    /// so that older values take precedence when there are conflicts.
    ///
    /// # Arguments
    ///
    /// * `provider` - Database provider for DB access and block lookups
    /// * `range` - Block range to accumulate reverts for (inclusive)
    ///
    /// # Returns
    ///
    /// Accumulated trie reverts for all blocks in the specified range
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Any block in the range is beyond the database tip
    /// - Database access fails
    /// - Block hash lookup fails
    /// - Changeset computation fails
    pub fn get_or_compute_range<P>(
        &self,
        provider: &P,
        range: RangeInclusive<BlockNumber>,
    ) -> ProviderResult<TrieUpdatesSorted>
    where
        P: DBProvider
            + StageCheckpointReader
            + ChangeSetReader
            + StorageChangeSetReader
            + BlockNumReader
            + StorageSettingsCache,
    {
        // Get the database tip block number
        let db_tip_block = provider
            .get_stage_checkpoint(reth_stages_types::StageId::Finish)?
            .as_ref()
            .map(|chk| chk.block_number)
            .ok_or_else(|| ProviderError::InsufficientChangesets {
                requested: *range.start(),
                available: 0..=0,
            })?;

        let start_block = *range.start();
        let end_block = *range.end();

        // If range end is beyond the tip, return an error
        if end_block > db_tip_block {
            return Err(ProviderError::InsufficientChangesets {
                requested: end_block,
                available: 0..=db_tip_block,
            });
        }

        let timer = Instant::now();

        debug!(
            target: "trie::changeset_cache",
            start_block,
            end_block,
            db_tip_block,
            "Starting get_or_compute_range"
        );

        // Use changeset cache to retrieve and accumulate reverts block by block.
        // Iterate in reverse order (newest to oldest) so that older changesets
        // take precedence when there are conflicting updates.
        let mut accumulated_reverts = TrieUpdatesSorted::default();

        // Maintain the cumulative *state* revert incrementally as the walk descends.
        //
        // The per-block changeset computation needs `from_reverts((block+1)..=db_tip)` — the
        // state as it was just after `block` was processed. Computing that from scratch for
        // every block makes the walk O(range²) in changeset-entry reads (the deceleration this
        // fix targets). Instead we start from the suffix above `end_block` and extend it by one
        // block's per-block revert on each descending step:
        //
        //   cumulative(block-1) == cumulative(block).extend(individual(block))
        //
        // which is exactly the `from_reverts((block)..=db_tip)` set (`extend_ref_and_sort` gives
        // the older block precedence — the same "oldest occurrence wins" rule the suffix read
        // uses), so the accumulated cumulative is byte-identical to the from-scratch read while
        // costing only one per-block revert read (O(1) amortized) per step → the walk is O(range).
        let mut cumulative_state_revert =
            crate::state::from_reverts_auto(provider, (end_block + 1)..)?;

        for block_number in range.rev() {
            // Get the block hash for this block number
            let block_hash = provider.block_hash(block_number)?.ok_or_else(|| {
                ProviderError::other(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("block hash not found for block number {}", block_number),
                ))
            })?;

            debug!(
                target: "trie::changeset_cache",
                block_number,
                ?block_hash,
                "Looked up block hash for block number in range"
            );

            // Per-block state revert for this block only (cheap: one block's changesets).
            let individual_state_revert =
                crate::state::from_reverts_auto(provider, block_number..=block_number)?;

            // Get changesets from cache (or compute on-the-fly using the incrementally
            // accumulated cumulative revert instead of a from-scratch suffix read).
            let changesets = self.get_or_compute_with(block_hash, block_number, || {
                compute_block_trie_changesets_from_reverts(
                    provider,
                    block_number,
                    &individual_state_revert,
                    &cumulative_state_revert,
                )
            })?;

            // Overlay this block's changesets on top of accumulated reverts.
            // Since we iterate newest to oldest, older values are added last
            // and overwrite any conflicting newer values (oldest changeset values take
            // precedence).
            accumulated_reverts.extend_ref_and_sort(&changesets);

            // Advance the cumulative state revert down to the next (older) block:
            // cumulative(block-1) = cumulative(block) + individual(block).
            cumulative_state_revert.extend_ref_and_sort(&individual_state_revert);
        }

        let elapsed = timer.elapsed();

        let num_account_nodes = accumulated_reverts.account_nodes_ref().len();
        let num_storage_tries = accumulated_reverts.storage_tries_ref().len();

        debug!(
            target: "trie::changeset_cache",
            ?elapsed,
            start_block,
            end_block,
            num_blocks = end_block.saturating_sub(start_block).saturating_add(1),
            num_account_nodes,
            num_storage_tries,
            "Finished accumulating trie reverts for block range"
        );

        Ok(accumulated_reverts)
    }
}

/// Guard for a pending changeset computation.
///
/// Returned by [`ChangesetCache::register_pending`]. Must be resolved via [`Self::resolve`]
/// to insert the computed changesets into the cache and wake waiting threads.
///
/// If dropped without resolving (e.g. due to a panic), the pending entry is automatically
/// cancelled so waiters fall through to the DB fallback.
#[must_use = "call .resolve() to insert changesets into the cache"]
pub struct PendingChangesetGuard {
    cache: ChangesetCache,
    block_hash: B256,
    /// `None` after [`Self::resolve`] has been called.
    pending: Option<Arc<PendingChangeset>>,
}

impl PendingChangesetGuard {
    /// Resolves the pending computation by inserting the changesets into the cache
    /// and waking all waiting threads.
    pub fn resolve(mut self, block_number: u64, changesets: Arc<TrieUpdatesSorted>) {
        self.cache.insert(self.block_hash, block_number, changesets);
        self.pending = None;
    }
}

impl fmt::Debug for PendingChangesetGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingChangesetGuard").field("block_hash", &self.block_hash).finish()
    }
}

impl Drop for PendingChangesetGuard {
    fn drop(&mut self) {
        let Some(pending) = self.pending.take() else {
            // Guard was resolved successfully already, no-op
            return
        };

        let removed = self.cache.inner.write().pending.remove(&self.block_hash);
        if let Some(removed) = removed {
            if Arc::ptr_eq(&removed, &pending) {
                debug!(
                    target: "trie::changeset_cache",
                    block_hash = ?self.block_hash,
                    "Pending changeset dropped without resolution, cancelling"
                );
                removed.cancel();
            } else {
                // Put it back — it belongs to a different registration.
                self.cache.inner.write().pending.insert(self.block_hash, removed);
            }
        }
    }
}

/// In-memory cache for trie changesets with explicit eviction policy.
///
/// Holds changesets for blocks that have been validated but not yet persisted.
/// Keyed by block hash for fast lookup during reorgs. Eviction is controlled
/// explicitly by the engine API tree handler when persistence completes.
///
/// ## Eviction Policy
///
/// Unlike traditional caches with automatic eviction, this cache requires explicit
/// eviction calls. The engine API tree handler calls `evict(block_number)` after
/// blocks are persisted to the database, ensuring changesets remain available
/// until their corresponding blocks are safely on disk.
///
/// ## Metrics
///
/// The cache maintains several metrics for observability:
/// - `hits`: Number of successful cache lookups
/// - `misses`: Number of failed cache lookups
/// - `evictions`: Number of blocks evicted
/// - `size`: Current number of cached blocks
#[derive(Debug)]
struct ChangesetCacheInner {
    /// Cache entries: block hash -> (block number, changesets)
    entries: B256Map<(u64, Arc<TrieUpdatesSorted>)>,

    /// Block number to hashes mapping for eviction
    block_numbers: BTreeMap<u64, Vec<B256>>,

    /// Pending changeset computations: block hash -> pending entry.
    /// Threads waiting on a pending entry will block until it's resolved.
    pending: B256Map<Arc<PendingChangeset>>,

    /// Metrics for monitoring cache behavior
    #[cfg(feature = "metrics")]
    metrics: ChangesetCacheMetrics,
}

#[cfg(feature = "metrics")]
/// Metrics for the changeset cache.
///
/// These metrics provide visibility into cache performance and help identify
/// potential issues like high miss rates.
#[derive(Metrics, Clone)]
#[metrics(scope = "trie.changeset_cache")]
struct ChangesetCacheMetrics {
    /// Cache hit counter
    hits: Counter,

    /// Cache miss counter
    misses: Counter,

    /// Eviction counter
    evictions: Counter,

    /// Current cache size (number of entries)
    size: Gauge,
}

impl Default for ChangesetCacheInner {
    fn default() -> Self {
        Self::new()
    }
}

impl ChangesetCacheInner {
    /// Creates a new empty changeset cache.
    ///
    /// The cache has no capacity limit and relies on explicit eviction
    /// via the `evict()` method to manage memory usage.
    fn new() -> Self {
        Self {
            entries: B256Map::default(),
            block_numbers: BTreeMap::new(),
            pending: B256Map::default(),
            #[cfg(feature = "metrics")]
            metrics: Default::default(),
        }
    }

    fn get(&self, block_hash: &B256) -> Option<Arc<TrieUpdatesSorted>> {
        match self.entries.get(block_hash) {
            Some((_, changesets)) => {
                #[cfg(feature = "metrics")]
                self.metrics.hits.increment(1);
                Some(Arc::clone(changesets))
            }
            None => {
                #[cfg(feature = "metrics")]
                self.metrics.misses.increment(1);
                None
            }
        }
    }

    fn insert(&mut self, block_hash: B256, block_number: u64, changesets: Arc<TrieUpdatesSorted>) {
        debug!(
            target: "trie::changeset_cache",
            ?block_hash,
            block_number,
            cache_size_before = self.entries.len(),
            "Inserting changeset into cache"
        );

        // Insert the entry
        self.entries.insert(block_hash, (block_number, changesets));

        // Add block hash to block_numbers mapping
        self.block_numbers.entry(block_number).or_default().push(block_hash);

        // Update size metric
        #[cfg(feature = "metrics")]
        self.metrics.size.set(self.entries.len() as f64);

        debug!(
            target: "trie::changeset_cache",
            ?block_hash,
            block_number,
            cache_size_after = self.entries.len(),
            "Changeset inserted into cache"
        );
    }

    fn evict(&mut self, up_to_block: BlockNumber) {
        debug!(
            target: "trie::changeset_cache",
            up_to_block,
            cache_size_before = self.entries.len(),
            "Starting cache eviction"
        );

        // Find all block numbers that should be evicted (< up_to_block)
        let blocks_to_evict: Vec<u64> =
            self.block_numbers.range(..up_to_block).map(|(num, _)| *num).collect();

        // Remove entries for each block number below threshold
        #[cfg(feature = "metrics")]
        let mut evicted_count = 0;
        #[cfg(not(feature = "metrics"))]
        let mut evicted_count = 0;

        for block_number in &blocks_to_evict {
            if let Some(hashes) = self.block_numbers.remove(block_number) {
                debug!(
                    target: "trie::changeset_cache",
                    block_number,
                    num_hashes = hashes.len(),
                    "Evicting block from cache"
                );
                for hash in hashes {
                    if self.entries.remove(&hash).is_some() {
                        evicted_count += 1;
                    }
                }
            }
        }

        debug!(
            target: "trie::changeset_cache",
            up_to_block,
            evicted_count,
            cache_size_after = self.entries.len(),
            "Finished cache eviction"
        );

        // Update metrics if we evicted anything
        #[cfg(feature = "metrics")]
        if evicted_count > 0 {
            self.metrics.evictions.increment(evicted_count as u64);
            self.metrics.size.set(self.entries.len() as f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{
        map::{B256Map, HashMap},
        Address, U256,
    };
    use reth_db_api::{
        models::{AccountBeforeTx, BlockNumberAddress},
        tables,
        transaction::DbTxMut,
    };
    use reth_primitives_traits::{Account, StorageEntry};
    use reth_provider::test_utils::create_test_provider_factory;

    // Helper function to create empty TrieUpdatesSorted for testing
    fn create_test_changesets() -> Arc<TrieUpdatesSorted> {
        Arc::new(TrieUpdatesSorted::new(vec![], B256Map::default()))
    }

    /// Differential test for the incremental cumulative-revert accumulation used by
    /// [`ChangesetCache::get_or_compute_range`].
    ///
    /// The range walk maintains the cumulative state revert `from_reverts((block+1)..=db_tip)`
    /// incrementally (extend by one block's per-block revert per descending step) instead of
    /// recomputing it from scratch for every block. This test seeds a multi-block changeset
    /// fixture with keys that overlap across blocks (exercising the "oldest occurrence wins"
    /// precedence) and asserts, at every height, that:
    ///
    /// 1. the incrementally accumulated cumulative revert is byte-identical to the from-scratch
    ///    `from_reverts_auto((block+1)..)` read, and
    /// 2. the resulting per-block changesets from the incremental path
    ///    ([`compute_block_trie_changesets_from_reverts`]) are byte-identical to the from-scratch
    ///    path ([`compute_block_trie_changesets`]).
    #[test]
    fn incremental_cumulative_revert_matches_from_scratch() {
        let factory = create_test_provider_factory();
        let provider = factory.provider_rw().unwrap();
        let tx = provider.tx_ref();

        let addr_a = Address::with_last_byte(1);
        let addr_b = Address::with_last_byte(2);
        let addr_c = Address::with_last_byte(3);

        let put_account = |block: u64, address: Address, nonce: u64| {
            tx.put::<tables::AccountChangeSets>(
                block,
                AccountBeforeTx {
                    address,
                    info: Some(Account { nonce, balance: U256::from(nonce), bytecode_hash: None }),
                },
            )
            .unwrap();
        };
        let put_storage = |block: u64, address: Address, slot: u64, value: u64| {
            tx.put::<tables::StorageChangeSets>(
                BlockNumberAddress((block, address)),
                StorageEntry { key: B256::from(U256::from(slot)), value: U256::from(value) },
            )
            .unwrap();
        };

        let db_tip: u64 = 5;

        // Block 1: addr_a (nonce 1), addr_b (nonce 1); storage a/slot1, b/slot2.
        put_account(1, addr_a, 1);
        put_account(1, addr_b, 1);
        put_storage(1, addr_a, 1, 100);
        put_storage(1, addr_b, 2, 200);

        // Block 2: addr_a changes again (overlap -> block 1's revert must win in cumulative).
        put_account(2, addr_a, 2);
        put_storage(2, addr_a, 1, 101); // same slot as block 1 -> oldest (block 1) wins
        put_storage(2, addr_c, 3, 300); // new slot/account

        // Block 3: addr_c account, addr_b storage overlap.
        put_account(3, addr_c, 3);
        put_storage(3, addr_b, 2, 201); // overlaps block 1's b/slot2

        // Block 4: no changes (empty block) — exercises empty per-block revert.

        // Block 5 (db tip): addr_a again, new storage.
        put_account(5, addr_a, 5);
        put_storage(5, addr_a, 4, 500);

        // Walk newest -> oldest exactly as get_or_compute_range does, maintaining the cumulative
        // revert incrementally, and compare against the from-scratch computation at each height.
        let mut cumulative =
            crate::state::from_reverts_auto(&*provider, (db_tip + 1)..).unwrap();

        for block in (1..=db_tip).rev() {
            // (1) The incrementally accumulated cumulative must equal the from-scratch suffix read.
            let expected_cumulative =
                crate::state::from_reverts_auto(&*provider, (block + 1)..).unwrap();
            assert_eq!(
                cumulative, expected_cumulative,
                "cumulative revert mismatch at block {block}"
            );

            let individual =
                crate::state::from_reverts_auto(&*provider, block..=block).unwrap();

            // (2) The per-block changesets from the incremental path must equal the from-scratch
            // path. Both run against the same (empty) trie tables in this fixture, so any
            // difference could only come from a divergent revert input.
            let from_scratch = compute_block_trie_changesets(&*provider, block).unwrap();
            let incremental = compute_block_trie_changesets_from_reverts(
                &*provider,
                block,
                &individual,
                &cumulative,
            )
            .unwrap();
            assert_eq!(
                from_scratch, incremental,
                "block trie changesets mismatch at block {block}"
            );

            // Advance the cumulative down to the next (older) block.
            cumulative.extend_ref_and_sort(&individual);
        }

        // After descending past block 1, the cumulative equals from_reverts(1..) = the full range.
        let full = crate::state::from_reverts_auto(&*provider, 1..).unwrap();
        assert_eq!(cumulative, full, "final cumulative should span the whole range");
    }

    #[test]
    fn test_insert_and_retrieve_single_entry() {
        let mut cache = ChangesetCacheInner::new();
        let hash = B256::random();
        let changesets = create_test_changesets();

        cache.insert(hash, 100, Arc::clone(&changesets));

        // Should be able to retrieve it
        let retrieved = cache.get(&hash);
        assert!(retrieved.is_some());
        assert_eq!(cache.entries.len(), 1);
    }

    #[test]
    fn test_insert_multiple_entries() {
        let mut cache = ChangesetCacheInner::new();

        // Insert 10 blocks
        let mut hashes = Vec::new();
        for i in 0..10 {
            let hash = B256::random();
            cache.insert(hash, 100 + i, create_test_changesets());
            hashes.push(hash);
        }

        // Should be able to retrieve all
        assert_eq!(cache.entries.len(), 10);
        for hash in &hashes {
            assert!(cache.get(hash).is_some());
        }
    }

    #[test]
    fn test_eviction_when_explicitly_called() {
        let mut cache = ChangesetCacheInner::new();

        // Insert 15 blocks (0-14)
        let mut hashes = Vec::new();
        for i in 0..15 {
            let hash = B256::random();
            cache.insert(hash, i, create_test_changesets());
            hashes.push((i, hash));
        }

        // All blocks should be present (no automatic eviction)
        assert_eq!(cache.entries.len(), 15);

        // Explicitly evict blocks < 4
        cache.evict(4);

        // Blocks 0-3 should be evicted
        assert_eq!(cache.entries.len(), 11); // blocks 4-14 = 11 blocks

        // Verify blocks 0-3 are evicted
        for i in 0..4 {
            assert!(cache.get(&hashes[i as usize].1).is_none(), "Block {} should be evicted", i);
        }

        // Verify blocks 4-14 are still present
        for i in 4..15 {
            assert!(cache.get(&hashes[i as usize].1).is_some(), "Block {} should be present", i);
        }
    }

    #[test]
    fn test_eviction_with_persistence_watermark() {
        let mut cache = ChangesetCacheInner::new();

        // Insert blocks 100-165
        let mut hashes = HashMap::new();
        for i in 100..=165 {
            let hash = B256::random();
            cache.insert(hash, i, create_test_changesets());
            hashes.insert(i, hash);
        }

        // All blocks should be present (no automatic eviction)
        assert_eq!(cache.entries.len(), 66);

        // Simulate persistence up to block 164, with 64-block retention window
        // Eviction threshold = 164 - 64 = 100
        cache.evict(100);

        // Blocks 100-165 should remain (66 blocks)
        assert_eq!(cache.entries.len(), 66);

        // Simulate persistence up to block 165
        // Eviction threshold = 165 - 64 = 101
        cache.evict(101);

        // Blocks 101-165 should remain (65 blocks)
        assert_eq!(cache.entries.len(), 65);
        assert!(cache.get(&hashes[&100]).is_none());
        assert!(cache.get(&hashes[&101]).is_some());
    }

    #[test]
    fn test_out_of_order_inserts_with_explicit_eviction() {
        let mut cache = ChangesetCacheInner::new();

        // Insert blocks in random order
        let hash_10 = B256::random();
        cache.insert(hash_10, 10, create_test_changesets());

        let hash_5 = B256::random();
        cache.insert(hash_5, 5, create_test_changesets());

        let hash_15 = B256::random();
        cache.insert(hash_15, 15, create_test_changesets());

        let hash_3 = B256::random();
        cache.insert(hash_3, 3, create_test_changesets());

        // All blocks should be present (no automatic eviction)
        assert_eq!(cache.entries.len(), 4);

        // Explicitly evict blocks < 5
        cache.evict(5);

        assert!(cache.get(&hash_3).is_none(), "Block 3 should be evicted");
        assert!(cache.get(&hash_5).is_some(), "Block 5 should be present");
        assert!(cache.get(&hash_10).is_some(), "Block 10 should be present");
        assert!(cache.get(&hash_15).is_some(), "Block 15 should be present");
    }

    #[test]
    fn test_multiple_blocks_same_number() {
        let mut cache = ChangesetCacheInner::new();

        // Insert multiple blocks with same number (side chains)
        let hash_1a = B256::random();
        let hash_1b = B256::random();
        cache.insert(hash_1a, 100, create_test_changesets());
        cache.insert(hash_1b, 100, create_test_changesets());

        // Both should be retrievable
        assert!(cache.get(&hash_1a).is_some());
        assert!(cache.get(&hash_1b).is_some());
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn test_eviction_removes_all_side_chains() {
        let mut cache = ChangesetCacheInner::new();

        // Insert multiple blocks at same height (side chains)
        let hash_10a = B256::random();
        let hash_10b = B256::random();
        let hash_10c = B256::random();
        cache.insert(hash_10a, 10, create_test_changesets());
        cache.insert(hash_10b, 10, create_test_changesets());
        cache.insert(hash_10c, 10, create_test_changesets());

        let hash_20 = B256::random();
        cache.insert(hash_20, 20, create_test_changesets());

        assert_eq!(cache.entries.len(), 4);

        // Evict blocks < 15 - should remove all three side chains at height 10
        cache.evict(15);

        assert_eq!(cache.entries.len(), 1);
        assert!(cache.get(&hash_10a).is_none());
        assert!(cache.get(&hash_10b).is_none());
        assert!(cache.get(&hash_10c).is_none());
        assert!(cache.get(&hash_20).is_some());
    }
}
