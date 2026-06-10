//! Engine orchestrator launch helper.
//!
//! Provides [`build_engine_orchestrator`](crate::launch::build_engine_orchestrator) which wires
//! together all engine components and returns a
//! [`ChainOrchestrator`](crate::chain::ChainOrchestrator) ready to be polled as a `Stream`.

use crate::{
    backfill::PipelineSync,
    chain::ChainOrchestrator,
    download::BasicBlockDownloader,
    engine::{EngineApiKind, EngineApiRequest, EngineApiRequestHandler, EngineHandler},
    persistence::PersistenceHandle,
    tree::{EngineApiTreeHandler, EngineValidator, TreeConfig, WaitForCaches},
};
use futures::Stream;
use reth_consensus::FullConsensus;
use reth_engine_primitives::BeaconEngineMessage;
use reth_evm::ConfigureEvm;
use reth_network_p2p::BlockClient;
use reth_payload_builder::PayloadBuilderHandle;
use reth_primitives_traits::NodePrimitives;
use reth_provider::{
    providers::{BlockchainProvider, ProviderNodeTypes},
    ProviderFactory,
};
use reth_prune::PrunerWithFactory;
use reth_stages_api::{MetricEventsSender, Pipeline};
use reth_tasks::Runtime;
use reth_trie_db::ChangesetCache;
use std::sync::Arc;

/// Process-global escrow for the engine-tree request sender.
///
/// The stock launcher consumes the [`ChainOrchestrator`] (and with it the only
/// route to the tree's `incoming` channel), so an external consensus driver
/// that imports pre-executed blocks via
/// [`EngineApiRequest::InsertExecutedBlock`] has no way to reach the tree.
/// [`build_engine_orchestrator`] deposits a clone of the sender here; the
/// embedding binary takes it once after launch.
///
/// Single-node-per-process by design: a second launch overwrites the deposit,
/// and `take` hands the sender to exactly one caller.
pub mod tree_sender_escrow {
    use std::{any::Any, sync::Mutex};

    static ESCROW: Mutex<Option<Box<dyn Any + Send>>> = Mutex::new(None);

    pub(super) fn deposit(sender: Box<dyn Any + Send>) {
        *ESCROW.lock().expect("tree sender escrow poisoned") = Some(sender);
    }

    /// Take the deposited sender, downcast to the node's concrete channel
    /// type. `None` if nothing was deposited, it was already taken, or the
    /// type does not match the deposited node's engine types.
    pub fn take<T: 'static>() -> Option<T> {
        ESCROW
            .lock()
            .expect("tree sender escrow poisoned")
            .take()
            .and_then(|boxed| boxed.downcast::<T>().ok())
            .map(|boxed| *boxed)
    }
}

/// Builds the engine [`ChainOrchestrator`] that drives the chain forward.
///
/// This spawns and wires together the following components:
///
/// - **[`BasicBlockDownloader`]** — downloads blocks on demand from the network during live sync.
/// - **[`PersistenceHandle`]** — spawns the persistence service on a background thread for writing
///   blocks and performing pruning outside the critical consensus path.
/// - **[`EngineApiTreeHandler`]** — spawns the tree handler that processes engine API requests
///   (`newPayload`, `forkchoiceUpdated`) and maintains the in-memory chain state.
/// - **[`EngineApiRequestHandler`]** + **[`EngineHandler`]** — glue that routes incoming CL
///   messages to the tree handler and manages download requests.
/// - **[`PipelineSync`]** — wraps the staged sync [`Pipeline`] for backfill sync when the node
///   needs to catch up over large block ranges.
///
/// The returned orchestrator implements [`Stream`] and yields
/// [`ChainEvent`]s.
///
/// [`ChainEvent`]: crate::chain::ChainEvent
#[expect(clippy::too_many_arguments, clippy::type_complexity)]
pub fn build_engine_orchestrator<N, Client, S, V, C>(
    engine_kind: EngineApiKind,
    consensus: Arc<dyn FullConsensus<N::Primitives>>,
    client: Client,
    incoming_requests: S,
    pipeline: Pipeline<N>,
    pipeline_task_spawner: Runtime,
    provider: ProviderFactory<N>,
    blockchain_db: BlockchainProvider<N>,
    pruner: PrunerWithFactory<ProviderFactory<N>>,
    payload_builder: PayloadBuilderHandle<N::Payload>,
    payload_validator: V,
    tree_config: TreeConfig,
    sync_metrics_tx: MetricEventsSender,
    evm_config: C,
    changeset_cache: ChangesetCache,
    runtime: Runtime,
) -> ChainOrchestrator<
    EngineHandler<
        EngineApiRequestHandler<EngineApiRequest<N::Payload, N::Primitives>, N::Primitives>,
        S,
        BasicBlockDownloader<Client, <N::Primitives as NodePrimitives>::Block>,
    >,
    PipelineSync<N>,
>
where
    N: ProviderNodeTypes,
    Client: BlockClient<Block = <N::Primitives as NodePrimitives>::Block> + 'static,
    S: Stream<Item = BeaconEngineMessage<N::Payload>> + Send + Sync + Unpin + 'static,
    V: EngineValidator<N::Payload> + WaitForCaches,
    C: ConfigureEvm<Primitives = N::Primitives> + 'static,
{
    let downloader = BasicBlockDownloader::new(client, consensus.clone());

    let persistence_handle =
        PersistenceHandle::<N::Primitives>::spawn_service(provider, pruner, sync_metrics_tx);

    let canonical_in_memory_state = blockchain_db.canonical_in_memory_state();

    let (to_tree_tx, from_tree) = EngineApiTreeHandler::spawn_new(
        blockchain_db,
        consensus,
        payload_validator,
        persistence_handle,
        payload_builder,
        canonical_in_memory_state,
        tree_config,
        engine_kind,
        evm_config,
        changeset_cache,
        runtime,
    );

    tree_sender_escrow::deposit(Box::new(to_tree_tx.clone()));
    let engine_handler = EngineApiRequestHandler::new(to_tree_tx, from_tree);
    let handler = EngineHandler::new(engine_handler, downloader, incoming_requests);

    let backfill_sync = PipelineSync::new(pipeline, pipeline_task_spawner);

    ChainOrchestrator::new(handler, backfill_sync)
}
