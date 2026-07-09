use alloy_primitives::U256;
use clap::{builder::Resettable, Args};
use reth_rpc_eth_types::GasPriceOracleConfig;
use reth_rpc_server_types::constants::gas_oracle::{
    DEFAULT_GAS_PRICE_BLOCKS, DEFAULT_GAS_PRICE_PERCENTILE, DEFAULT_MAX_GAS_PRICE,
};

/// Default gas price below which GPO ignores transactions.
const DEFAULT_IGNORE_GAS_PRICE: u64 = 1_000_000;
/// Default gas price to use if there are no blocks available.
const DEFAULT_SUGGESTED_GAS_PRICE: u64 = 1_000_000;

/// Parameters to configure Gas Price Oracle
#[derive(Debug, Clone, Copy, Args, PartialEq, Eq)]
#[command(next_help_heading = "Gas Price Oracle")]
pub struct GasPriceOracleArgs {
    /// Number of recent blocks to check for gas price
    #[arg(long = "gpo.blocks", default_value_t = DEFAULT_GAS_PRICE_BLOCKS)]
    pub blocks: u32,

    /// Gas Price below which gpo will ignore transactions
    #[arg(long = "gpo.ignoreprice", default_value_t = DEFAULT_IGNORE_GAS_PRICE)]
    pub ignore_price: u64,

    /// Maximum transaction priority fee(or gasprice before London Fork) to be recommended by gpo
    #[arg(long = "gpo.maxprice", default_value_t = DEFAULT_MAX_GAS_PRICE.to())]
    pub max_price: u64,

    /// The percentile of gas prices to use for the estimate
    #[arg(long = "gpo.percentile", default_value_t = DEFAULT_GAS_PRICE_PERCENTILE)]
    pub percentile: u32,

    /// The default gas price to use if there are no blocks to use
    #[arg(long = "gpo.default-suggested-fee", default_value = Resettable::from(Some(DEFAULT_SUGGESTED_GAS_PRICE.to_string().into())))]
    pub default_suggested_fee: Option<U256>,
}

impl GasPriceOracleArgs {
    /// Returns a [`GasPriceOracleConfig`] from the arguments.
    pub fn gas_price_oracle_config(&self) -> GasPriceOracleConfig {
        let Self { blocks, ignore_price, max_price, percentile, default_suggested_fee } = self;
        GasPriceOracleConfig {
            max_price: Some(U256::from(*max_price)),
            ignore_price: Some(U256::from(*ignore_price)),
            percentile: *percentile,
            blocks: *blocks,
            default_suggested_fee: *default_suggested_fee,
            ..Default::default()
        }
    }
}

impl Default for GasPriceOracleArgs {
    fn default() -> Self {
        Self {
            blocks: DEFAULT_GAS_PRICE_BLOCKS,
            ignore_price: DEFAULT_IGNORE_GAS_PRICE,
            max_price: DEFAULT_MAX_GAS_PRICE.to(),
            percentile: DEFAULT_GAS_PRICE_PERCENTILE,
            default_suggested_fee: Some(U256::from(DEFAULT_SUGGESTED_GAS_PRICE)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    /// A helper type to parse Args more easily
    #[derive(Parser)]
    struct CommandParser<T: Args> {
        #[command(flatten)]
        args: T,
    }

    #[test]
    fn test_parse_gpo_args() {
        let args = CommandParser::<GasPriceOracleArgs>::parse_from(["reth"]).args;
        assert_eq!(
            args,
            GasPriceOracleArgs {
                blocks: DEFAULT_GAS_PRICE_BLOCKS,
                ignore_price: DEFAULT_IGNORE_GAS_PRICE,
                max_price: DEFAULT_MAX_GAS_PRICE.to(),
                percentile: DEFAULT_GAS_PRICE_PERCENTILE,
                default_suggested_fee: Some(U256::from(DEFAULT_SUGGESTED_GAS_PRICE)),
            }
        );
    }

    #[test]
    fn gpo_args_default_sanity_test() {
        let default_args = GasPriceOracleArgs::default();
        let args = CommandParser::<GasPriceOracleArgs>::parse_from(["reth"]).args;
        assert_eq!(args, default_args);
    }

    #[test]
    fn gpo_args_use_requested_defaults_and_allow_overrides() {
        let args = CommandParser::<GasPriceOracleArgs>::parse_from(["reth"]).args;
        assert_eq!(args.ignore_price, DEFAULT_IGNORE_GAS_PRICE);
        assert_eq!(args.default_suggested_fee, Some(U256::from(DEFAULT_SUGGESTED_GAS_PRICE)));

        let args = CommandParser::<GasPriceOracleArgs>::parse_from([
            "reth",
            "--gpo.ignoreprice",
            "42",
            "--gpo.default-suggested-fee",
            "43",
        ])
        .args;
        assert_eq!(args.ignore_price, 42);
        assert_eq!(args.default_suggested_fee, Some(U256::from(43)));
    }
}
