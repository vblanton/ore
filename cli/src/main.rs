use std::{collections::HashMap, str::FromStr};
use std::time::{Instant, Duration};

use entropy_api::prelude::*;
use jup_swap::{
    quote::QuoteRequest,
    swap::SwapRequest,
    transaction_config::{DynamicSlippageSettings, TransactionConfig},
    JupiterSwapApiClient,
};
use ore_api::prelude::*;
use solana_account_decoder::UiAccountEncoding;
use solana_client::{
    client_error::{reqwest::StatusCode, ClientErrorKind},
    nonblocking::rpc_client::RpcClient,
    nonblocking::pubsub_client::PubsubClient,
    rpc_config::{RpcAccountInfoConfig, RpcProgramAccountsConfig},
    rpc_filter::{Memcmp, RpcFilterType},
    rpc_response::SlotInfo,
};
use futures_util::StreamExt;
use tokio::sync::watch;
use tokio::sync::RwLock;
use std::sync::Arc;
use solana_account_decoder::UiAccountData;
use solana_sdk::hash::Hash;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use solana_sdk::{
    address_lookup_table::{state::AddressLookupTable, AddressLookupTableAccount},
    commitment_config::CommitmentConfig,
    compute_budget::ComputeBudgetInstruction,
    message::{v0::Message, VersionedMessage},
    native_token::{lamports_to_sol, LAMPORTS_PER_SOL},
    pubkey::Pubkey,
    signature::{read_keypair_file, Signature, Signer},
    transaction::{Transaction, VersionedTransaction},
};
use solana_sdk::{keccak, pubkey};
use spl_associated_token_account::get_associated_token_address;
use spl_token::amount_to_ui_amount;
use steel::{AccountDeserialize, Clock, Discriminator, Instruction};

// Decode UiAccountData into raw bytes (Base64)
fn decode_ui_account_data(data: &UiAccountData) -> Option<Vec<u8>> {
    match data {
        UiAccountData::Binary(b64, _enc) => BASE64.decode(b64).ok(),
        UiAccountData::LegacyBinary(b64) => BASE64.decode(b64).ok(),
        _ => None,
    }
}

// Subscribe to an account and stream decoded state T via watch channel
pub async fn start_account_watch<T>(
    ws_url: &str,
    key: Pubkey,
) -> Result<watch::Receiver<Option<T>>, anyhow::Error>
where
    T: AccountDeserialize + Clone + Send + Sync + 'static,
{
    let (tx, rx) = watch::channel::<Option<T>>(None);
    let ws_url = ws_url.to_string();
    tokio::spawn(async move {
        loop {
            match PubsubClient::new(&ws_url).await {
                Ok(client) => {
                    let cfg = RpcAccountInfoConfig {
                        encoding: Some(UiAccountEncoding::Base64),
                        ..Default::default()
                    };
                    match client.account_subscribe(&key, Some(cfg)).await {
                        Ok((mut stream, _unsub)) => {
                            while let Some(update) = stream.next().await {
                                if let Some(bytes) = decode_ui_account_data(&update.value.data) {
                                    if let Ok(parsed) = T::try_from_bytes(&bytes) {
                                        let _ = tx.send(Some(parsed.clone()));
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            tokio::time::sleep(Duration::from_millis(300)).await;
                        }
                    }
                }
                Err(_) => {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        }
    });
    Ok(rx)
}

#[derive(Clone)]
struct BlockhashCache {
    inner: Arc<RwLock<Option<(Hash, u64)>>>,
    max_age_slots: u64,
    safety_margin_slots: u64,
}

impl BlockhashCache {
    fn new(max_age_slots: u64, safety_margin_slots: u64) -> Self {
        Self {
            inner: Arc::new(RwLock::new(None)),
            max_age_slots,
            safety_margin_slots,
        }
    }

    async fn get_fresh(
        &self,
        rpc: &RpcClient,
        current_slot_opt: Option<u64>,
    ) -> Result<Hash, anyhow::Error> {
        let now_slot = match current_slot_opt {
            Some(s) => s,
            None => rpc.get_slot().await?,
        };

        // Try cached if sufficiently fresh
        if let Some((h, at_slot)) = *self.inner.read().await {
            let age = now_slot.saturating_sub(at_slot);
            if age <= self.max_age_slots.saturating_sub(self.safety_margin_slots) {
                return Ok(h);
            }
            // Near expiry: verify with RPC before reuse
            if age <= self.max_age_slots {
                if rpc.is_blockhash_valid(&h, CommitmentConfig::processed()).await.unwrap_or(false) {
                    return Ok(h);
                }
            }
        }

        // Refresh
        let fresh = rpc.get_latest_blockhash().await?;
        {
            let mut w = self.inner.write().await;
            *w = Some((fresh, now_slot));
        }
        Ok(fresh)
    }

    #[allow(dead_code)]
    async fn invalidate(&self) {
        let mut w = self.inner.write().await;
        *w = None;
    }
}

#[tokio::main]
async fn main() {
    // Auto-load .env if present
    let _ = dotenvy::dotenv();

    // Load keypair from file or base58/json env
    let payer = load_keypair();

    // Build RPC with processed commitment (aligns freshness with official frontend)
    let rpc = RpcClient::new_with_commitment(
        std::env::var("RPC").expect("Missing RPC env var"),
        CommitmentConfig::processed(),
    );
    match std::env::var("COMMAND")
        .expect("Missing COMMAND env var")
        .as_str()
    {
        "auto_ev" => {
            auto_ev(&rpc, &payer).await.unwrap();
        }
        "timeleft" => {
            timeleft(&rpc).await.unwrap();
        }
        "automations" => {
            log_automations(&rpc).await.unwrap();
        }
        "clock" => {
            log_clock(&rpc).await.unwrap();
        }
        "claim" => {
            claim(&rpc, &payer).await.unwrap();
        }
        "board" => {
            log_board(&rpc).await.unwrap();
        }
        "config" => {
            log_config(&rpc).await.unwrap();
        }
        "bury" => {
            bury(&rpc, &payer).await.unwrap();
        }
        "reset" => {
            reset(&rpc, &payer).await.unwrap();
        }
        "treasury" => {
            log_treasury(&rpc).await.unwrap();
        }
        "miner" => {
            log_miner(&rpc, &payer).await.unwrap();
        }
        // "pool" => {
        //     log_meteora_pool(&rpc).await.unwrap();
        // }
        "deploy" => {
            deploy(&rpc, &payer).await.unwrap();
        }
        "stake" => {
            log_stake(&rpc, &payer).await.unwrap();
        }
        "deploy_all" => {
            deploy_all(&rpc, &payer).await.unwrap();
        }
        "round" => {
            log_round(&rpc).await.unwrap();
        }
        "set_admin" => {
            set_admin(&rpc, &payer).await.unwrap();
        }
        "set_fee_collector" => {
            set_fee_collector(&rpc, &payer).await.unwrap();
        }
        "ata" => {
            ata(&rpc, &payer).await.unwrap();
        }
        "checkpoint" => {
            checkpoint(&rpc, &payer).await.unwrap();
        }
        "checkpoint_all" => {
            checkpoint_all(&rpc, &payer).await.unwrap();
        }
        "close_all" => {
            close_all(&rpc, &payer).await.unwrap();
        }
        "participating_miners" => {
            participating_miners(&rpc).await.unwrap();
        }
        "new_var" => {
            new_var(&rpc, &payer).await.unwrap();
        }
        "set_buffer" => {
            set_buffer(&rpc, &payer).await.unwrap();
        }
        "set_swap_program" => {
            set_swap_program(&rpc, &payer).await.unwrap();
        }
        "keys" => {
            keys().await.unwrap();
        }
		"swap_sim" => {
			swap_sim(&rpc, &payer).await.unwrap();
		}
        "backtest" => {
            backtest(&rpc, &payer).await.unwrap();
        }
        "price" => {
            price(&rpc, &payer).await.unwrap();
        }
        _ => panic!("Invalid command"),
    };
}

fn load_keypair() -> solana_sdk::signer::keypair::Keypair {
    if let Ok(b58) = std::env::var("KEYPAIR_BASE58") {
        if !b58.trim().is_empty() {
            let bytes = bs58::decode(b58.trim()).into_vec().expect("Invalid KEYPAIR_BASE58");
            return solana_sdk::signer::keypair::Keypair::from_bytes(&bytes)
                .expect("Invalid secret key bytes in KEYPAIR_BASE58");
        }
    }
    if let Ok(json) = std::env::var("KEYPAIR_JSON") {
        if !json.trim().is_empty() {
            let arr: Vec<u8> = serde_json::from_str(&json).expect("Invalid KEYPAIR_JSON array");
            return solana_sdk::signer::keypair::Keypair::from_bytes(&arr)
                .expect("Invalid secret key bytes in KEYPAIR_JSON");
        }
    }
    let path = std::env::var("KEYPAIR").expect("Missing KEYPAIR env var");
    read_keypair_file(&path).expect("Failed to read keypair file")
}

async fn set_buffer(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let buffer = std::env::var("BUFFER").expect("Missing BUFFER env var");
    let buffer = u64::from_str(&buffer).expect("Invalid BUFFER");
    let ix = ore_api::sdk::set_buffer(payer.pubkey(), buffer);
    submit_transaction(rpc, payer, &[ix]).await?;
    Ok(())
}

async fn new_var(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let provider = std::env::var("PROVIDER").expect("Missing PROVIDER env var");
    let provider = Pubkey::from_str(&provider).expect("Invalid PROVIDER");
    let commit = std::env::var("COMMIT").expect("Missing COMMIT env var");
    let commit = keccak::Hash::from_str(&commit).expect("Invalid COMMIT");
    let samples = std::env::var("SAMPLES").expect("Missing SAMPLES env var");
    let samples = u64::from_str(&samples).expect("Invalid SAMPLES");
    let board_address = board_pda().0;
    let var_address = entropy_api::state::var_pda(board_address, 0).0;
    println!("Var address: {}", var_address);
    let ix = ore_api::sdk::new_var(payer.pubkey(), provider, 0, commit.to_bytes(), samples);
    submit_transaction(rpc, payer, &[ix]).await?;
    Ok(())
}

async fn participating_miners(rpc: &RpcClient) -> Result<(), anyhow::Error> {
    let round_id = std::env::var("ID").expect("Missing ID env var");
    let round_id = u64::from_str(&round_id).expect("Invalid ID");
    let miners = get_miners_participating(rpc, round_id).await?;
    for (i, (_address, miner)) in miners.iter().enumerate() {
        println!("{}: {}", i, miner.authority);
    }
    Ok(())
}

async fn log_stake(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let authority = std::env::var("AUTHORITY").unwrap_or(payer.pubkey().to_string());
    let authority = Pubkey::from_str(&authority).expect("Invalid AUTHORITY");
    let staker_address = ore_api::state::stake_pda(authority).0;
    let stake = get_stake(rpc, authority).await?;
    println!("Stake");
    println!("  address: {}", staker_address);
    println!("  authority: {}", authority);
    println!(
        "  balance: {} ORE",
        amount_to_ui_amount(stake.balance, TOKEN_DECIMALS)
    );
    println!("  last_claim_at: {}", stake.last_claim_at);
    println!("  last_deposit_at: {}", stake.last_deposit_at);
    println!("  last_withdraw_at: {}", stake.last_withdraw_at);
    println!(
        "  rewards_factor: {}",
        stake.rewards_factor.to_i80f48().to_string()
    );
    println!(
        "  rewards: {} ORE",
        amount_to_ui_amount(stake.rewards, TOKEN_DECIMALS)
    );
    println!(
        "  lifetime_rewards: {} ORE",
        amount_to_ui_amount(stake.lifetime_rewards, TOKEN_DECIMALS)
    );

    Ok(())
}

async fn ata(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let user = pubkey!("FgZFnb3bi7QexKCdXWPwWy91eocUD7JCFySHb83vLoPD");
    let token = pubkey!("8H8rPiWW4iTFCfEkSnf7jpqeNpFfvdH9gLouAL3Fe2Zx");
    let ata = get_associated_token_address(&user, &token);
    let ix = spl_associated_token_account::instruction::create_associated_token_account(
        &payer.pubkey(),
        &user,
        &token,
        &spl_token::ID,
    );
    submit_transaction(rpc, payer, &[ix]).await?;
    let account = rpc.get_account(&ata).await?;
    println!("ATA: {}", ata);
    println!("Account: {:?}", account);
    Ok(())
}

async fn keys() -> Result<(), anyhow::Error> {
    let treasury_address = ore_api::state::treasury_pda().0;
    let config_address = ore_api::state::config_pda().0;
    let board_address = ore_api::state::board_pda().0;
    let address = pubkey!("pqspJ298ryBjazPAr95J9sULCVpZe3HbZTWkbC1zrkS");
    let miner_address = ore_api::state::miner_pda(address).0;
    let round = round_pda(31460).0;
    println!("Round: {}", round);
    println!("Treasury: {}", treasury_address);
    println!("Config: {}", config_address);
    println!("Board: {}", board_address);
    println!("Miner: {}", miner_address);
    Ok(())
}

async fn claim(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let ix_sol = ore_api::sdk::claim_sol(payer.pubkey());
    let ix_ore = ore_api::sdk::claim_ore(payer.pubkey());
    submit_transaction(rpc, payer, &[ix_sol, ix_ore]).await?;
    Ok(())
}

async fn bury(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    // Get swap amount.
    let treasury = get_treasury(rpc).await?;
    let amount = treasury.balance.min(10 * LAMPORTS_PER_SOL);

    // Build quote request.
    const INPUT_MINT: Pubkey = pubkey!("So11111111111111111111111111111111111111112");
    const OUTPUT_MINT: Pubkey = pubkey!("oreoU2P8bN6jkk3jbaiVxYnG1dCXcYxwhwyK9jSybcp");
    let api_base_url =
        std::env::var("API_BASE_URL").unwrap_or("https://lite-api.jup.ag/swap/v1".into());
    let jupiter_swap_api_client = JupiterSwapApiClient::new(api_base_url);
    let quote_request = QuoteRequest {
        amount,
        input_mint: INPUT_MINT,
        output_mint: OUTPUT_MINT,
        max_accounts: Some(55),
        ..QuoteRequest::default()
    };

    // GET /quote
    let quote_response = match jupiter_swap_api_client.quote(&quote_request).await {
        Ok(quote_response) => quote_response,
        Err(e) => {
            println!("quote failed: {e:#?}");
            return Err(anyhow::anyhow!("quote failed: {e:#?}"));
        }
    };

    // GET /swap/instructions
    let treasury_address = ore_api::state::treasury_pda().0;
    let response = jupiter_swap_api_client
        .swap_instructions(&SwapRequest {
            user_public_key: treasury_address,
            quote_response,
            config: TransactionConfig {
                skip_user_accounts_rpc_calls: true,
                wrap_and_unwrap_sol: false,
                dynamic_compute_unit_limit: true,
                dynamic_slippage: Some(DynamicSlippageSettings {
                    min_bps: Some(50),
                    max_bps: Some(1000),
                }),
                ..TransactionConfig::default()
            },
        })
        .await
        .unwrap();

    let address_lookup_table_accounts =
        get_address_lookup_table_accounts(rpc, response.address_lookup_table_addresses)
            .await
            .unwrap();

    // Build transaction.
    let wrap_ix = ore_api::sdk::wrap(payer.pubkey());
    let bury_ix = ore_api::sdk::bury(
        payer.pubkey(),
        &response.swap_instruction.accounts,
        &response.swap_instruction.data,
    );
    simulate_transaction_with_address_lookup_tables(
        rpc,
        payer,
        &[wrap_ix, bury_ix],
        address_lookup_table_accounts,
    )
    .await;

    Ok(())
}

async fn auto_ev(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    // Config (env or defaults)
    let price_api_base = std::env::var("API_BASE_URL").unwrap_or("https://lite-api.jup.ag/swap/v1".into());
    let jupiter = JupiterSwapApiClient::new(price_api_base);
    let snipe_slots: u64 = std::env::var("AUTO_SNIPE_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let min_buffer_sol: f64 = std::env::var("AUTO_MIN_SOL_BUFFER_SOL").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);
    let min_budget_pct: f64 = std::env::var("AUTO_MIN_BUDGET_PCT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.005);
    let max_budget_pct: f64 = std::env::var("AUTO_MAX_BUDGET_PCT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.02);
    let edge_scale: f64 = std::env::var("AUTO_EDGE_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let verbose: bool = std::env::var("AUTO_VERBOSE").ok().map(|v| v != "0").unwrap_or(false);
    let dry_run: bool = std::env::var("AUTO_DRY_RUN").ok().map(|v| v != "0").unwrap_or(false);
    let debug: bool = std::env::var("AUTO_DEBUG").ok().map(|v| v != "0").unwrap_or(false);
    let summary: bool = std::env::var("AUTO_SUMMARY").ok().map(|v| v != "0").unwrap_or(false);
    let max_squares: usize = std::env::var("AUTO_MAX_SQUARES").ok().and_then(|v| v.parse().ok()).unwrap_or(25);
    let min_per_square_lamports: u64 = std::env::var("AUTO_MIN_PER_SQUARE_LAMPORTS").ok().and_then(|v| v.parse().ok()).unwrap_or(10_000); // 0.00001 SOL
    let ignore_motherlode: bool = std::env::var("AUTO_IGNORE_MOTHERLODE").ok().map(|v| v != "0").unwrap_or(false);
    let ore_haircut: f64 = std::env::var("AUTO_ORE_HAIRCUT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.9);

    // No-op at startup; automation will be created/updated right before deploy

    // Track last played round for post-finalization reporting (round_id, mask, per, sel)
    let mut last_played: Option<(u64, [bool; 25], u64, usize)> = None;
    let mut logged_predeploy_checked_round: Option<u64> = None;
    // Summary tracking
    let start_sol_balance = rpc.get_balance(&payer.pubkey()).await.unwrap_or(0);
    let mut start_round_id: Option<u64> = None;
    let mut rounds_entered: u64 = 0;
    // Cached price with TTL
    let mut price_cache: Option<(f64, Instant)> = None;

    // Websocket slot feed
    let ws = std::env::var("WS").unwrap_or_else(|_| http_to_ws(&std::env::var("RPC").expect("Missing RPC env var")));
    let slot_rx = start_slot_watch(ws.clone()).await?;

    // Account watchers (prefer WS updates over HTTP)
    let board_key = ore_api::state::board_pda().0;
    let treasury_key = ore_api::state::treasury_pda().0;
    let miner_key = ore_api::state::miner_pda(payer.pubkey()).0;
    let mut board_rx = start_account_watch::<Board>(&ws, board_key).await?;
    let treasury_rx = start_account_watch::<Treasury>(&ws, treasury_key).await?;
    let _miner_rx = start_account_watch::<Miner>(&ws, miner_key).await?;

    // Blockhash cache with safety margins (env-tunable)
    let bh_max_age_slots = std::env::var("BH_MAX_AGE_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(75);
    let bh_safety_slots = std::env::var("BH_SAFETY_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let bh_cache = BlockhashCache::new(bh_max_age_slots, bh_safety_slots);

    // One-time startup cleanup: close existing Automation (if any) to avoid overriding manual deploys
    let automation_addr_start = ore_api::state::automation_pda(payer.pubkey()).0;
    if let Ok(acct) = rpc.get_account(&automation_addr_start).await {
        if !acct.data.is_empty() && ore_api::state::Automation::try_from_bytes(&acct.data).is_ok() {
            let close_ix = ore_api::sdk::automate(
                payer.pubkey(),
                0,
                0,
                Pubkey::default(), // executor == default => close automation
                0,
                0,
                ore_api::state::AutomationStrategy::Preferred as u8,
            );
            if !dry_run {
                let _ = submit_transaction(rpc, payer, &[close_ix]).await;
            }
        }
    }

    loop {
        // Load board (prefer WS; fallback HTTP)
        let board = if let Some(b) = board_rx.borrow().clone() {
            b
        } else {
            match get_board(rpc).await {
                Ok(b) => b,
                Err(e) => {
                    println!("auto_ev: failed to get board: {e:#?}");
                    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                    continue;
                }
            }
        };
        if start_round_id.is_none() { start_round_id = Some(board.round_id); }
        // Choose current slot from websocket; fallback to RPC only until WS delivers
        let mut current_slot = *slot_rx.borrow();
        if current_slot == 0 {
            current_slot = match get_current_slot(rpc).await {
                Ok(s) => s,
                Err(e) => {
                    println!("auto_ev: failed to get current slot: {e:#?}");
                    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                    continue;
                }
            };
        }
        // If round not started yet, wait
        if board.end_slot == u64::MAX {
            tokio::time::sleep(std::time::Duration::from_millis(800)).await;
            continue;
        }

        // Ensure previous round is checkpointed before any new deploys (log only; bundle later)
        if let Ok(m) = get_miner(rpc, payer.pubkey()).await {
            let prev_round_pending = m.checkpoint_id < m.round_id && m.round_id < board.round_id;
            if debug && prev_round_pending && logged_predeploy_checked_round != Some(m.round_id) {
                println!(
                    "pre-deploy checkpoint pending: round_id={} checkpoint_id={} rewards sol={} ore={}",
                    m.round_id, m.checkpoint_id, fmt_sol(m.rewards_sol), fmt_ore(m.rewards_ore)
                );
                logged_predeploy_checked_round = Some(m.round_id);
            }
        }

        // Sleep until inside snipe window (based on slots left)
        let slots_remaining = board.end_slot.saturating_sub(current_slot);
        if slots_remaining > snipe_slots {
            if debug {
                println!(
                    "R={} waiting slots_left={} (snipe={} slots)",
                    board.round_id,
                    slots_remaining,
                    snipe_slots
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            continue;
        }

        // Fetch latest round/treasury and price
        let round = match get_round(rpc, board.round_id).await {
            Ok(r) => r,
            Err(e) => {
                println!("auto_ev: failed to get round {}: {e:#?}", board.round_id);
                tokio::time::sleep(std::time::Duration::from_millis(400)).await;
                continue;
            }
        };
        let treasury = if let Some(t) = treasury_rx.borrow().clone() {
            t
        } else {
            match get_treasury(rpc).await {
                Ok(t) => t,
                Err(e) => {
                    println!("auto_ev: failed to get treasury: {e:#?}");
                    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
                    continue;
                }
            }
        };

        // Wallet balances
        let sol_balance = rpc.get_balance(&payer.pubkey()).await.unwrap_or(0);
        let buffer_target_lamports = (min_buffer_sol * LAMPORTS_PER_SOL as f64) as u64;
        let spendable = sol_balance.saturating_sub(buffer_target_lamports);
        let mut deploy_budget = spendable; // will be refined after computing edge
        if deploy_budget < min_per_square_lamports {
            // Ensure buffer; optionally top-up by selling ORE if desired (handled after round)
            tokio::time::sleep(std::time::Duration::from_millis(600)).await;
            continue;
        }

        // Get ORE->SOL price (per ORE) via 5s TTL cache
        let mut price_p_sol_per_ore = get_price_cached(&jupiter, &mut price_cache, 5_000).await;

        // Build EV at unit amount for candidates
        let total_deployed: u64 = round.deployed.iter().sum();
        let mut unit_ev_pairs: Vec<(usize, f64)> = Vec::new();
        let try_amount = min_per_square_lamports as f64; // unit
        let motherlode_now = if ignore_motherlode { 0.0 } else { treasury.motherlode as f64 / ONE_ORE as f64 };
        for i in 0..25 {
            let d_i = round.deployed[i] as f64;
            let total = total_deployed as f64;
            let ev_total_lamports = ev_total(try_amount, d_i, total, motherlode_now, price_p_sol_per_ore, ore_haircut);
            if ev_total_lamports > 0.0 { unit_ev_pairs.push((i, ev_total_lamports)); }
        }
        unit_ev_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if unit_ev_pairs.is_empty() {
            // Nothing profitable; skip this round
            if verbose {
                if debug {
                    // Compute diagnostics: cached price and age, best unit EV (even if negative), and EV for top-1 set
                            let (best_idx_any, best_unit_ev_all) = {
                        let mut best_i = 0usize;
                        let mut best_ev = f64::NEG_INFINITY;
                        for i in 0..25 {
                            let d_i = round.deployed[i] as f64;
                            let total = total_deployed as f64;
                            let ev_total = ev_total(try_amount, d_i, total, motherlode_now, price_p_sol_per_ore, ore_haircut);
                            if ev_total > best_ev {
                                best_ev = ev_total;
                                best_i = i;
                            }
                        }
                        (best_i, best_ev)
                    };
                    let ev1_lamports = ev_set_total_sol_lamports(
                        min_per_square_lamports,
                        &[best_idx_any],
                        &round,
                        total_deployed,
                        price_p_sol_per_ore,
                        motherlode_now,
                        ore_haircut,
                    );
                    let price_age_ms = price_cache
                        .as_ref()
                        .map(|(_, ts)| Instant::now().saturating_duration_since(*ts).as_millis() as u64)
                        .unwrap_or(0);
                    println!(
                        "R={} no profitable squares found price={:.6} age_ms={} best_unit_ev={:.3e} ev1_set≈{:.6} SOL (square={})",
                        board.round_id,
                        price_p_sol_per_ore,
                        price_age_ms,
                        best_unit_ev_all,
                        ev1_lamports / LAMPORTS_PER_SOL as f64,
                        best_idx_any
                    );
                } else {
                    println!("R={} no profitable squares found", board.round_id);
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        } else {
            let selected = std::cmp::min(max_squares, unit_ev_pairs.len());
            let candidates: Vec<usize> = unit_ev_pairs.iter().take(selected).map(|(i, _)| *i).collect();

            // Edge-adaptive budget sizing: scale by avg edge per lamport
            let avg_edge_per_lamport = unit_ev_pairs.iter().take(selected)
                .map(|(_, ev)| ev / try_amount)
                .sum::<f64>() / (selected as f64);
            let budget_pct = (avg_edge_per_lamport * edge_scale).clamp(min_budget_pct, max_budget_pct);
            let budget_lamports = ((sol_balance as f64) * budget_pct) as u64;
            deploy_budget = spendable.min(budget_lamports);

            let mut mask = [false; 25];
            for i in candidates.iter().copied() { mask[i] = true; }
            let count = candidates.len().max(1);
            let per_square_cap = (deploy_budget / (count as u64)).max(min_per_square_lamports) as f64;

            // Compute per-square profitable limits and choose common amount
            let mut per_square_limits: Vec<f64> = Vec::new();
                    for &i in candidates.iter() {
                let d_i = round.deployed[i] as f64;
                let total = total_deployed as f64;
                        let limit = max_profitable_amount(try_amount, per_square_cap, d_i, total, motherlode_now, price_p_sol_per_ore, ore_haircut);
                if limit >= try_amount { per_square_limits.push(limit); }
            }
            if per_square_limits.len() < count {
                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            } else {
                let common_amount = per_square_limits.into_iter().fold(per_square_cap, |m, x| m.min(x));
                let common_lamports = common_amount.floor() as u64;
                if common_lamports >= min_per_square_lamports {
                    // Dynamic candidate capping with interaction-aware set EV
                    let ordered = candidates.clone();
                    let mut set: Vec<usize> = Vec::new();
                    for &i in ordered.iter() {
                        if set.len() >= max_squares { break; }
                        let mut trial = set.clone();
                        trial.push(i);
                        let ev_before = ev_set_total_sol_lamports(
                            common_lamports, &set, &round, total_deployed, price_p_sol_per_ore, motherlode_now, ore_haircut
                        );
                        let ev_after  = ev_set_total_sol_lamports(
                            common_lamports, &trial, &round, total_deployed, price_p_sol_per_ore, motherlode_now, ore_haircut
                        );
                        if ev_after > ev_before {
                            set = trial;
                        } else {
                            // Kelly-aware tiebreak: if EV is effectively equal, accept when Kelly fraction improves
                            let eps_lamports = (1e-6f64) * (LAMPORTS_PER_SOL as f64); // ~0.000001 SOL tolerance
                            if ev_after >= ev_before - eps_lamports {
                                let f_before = kelly_fraction_from_outcomes(
                                    &set_outcomes_total_sol_lamports(
                                        common_lamports, &set, &round, total_deployed, price_p_sol_per_ore, motherlode_now, ore_haircut,
                                    )
                                );
                                let f_after = kelly_fraction_from_outcomes(
                                    &set_outcomes_total_sol_lamports(
                                        common_lamports, &trial, &round, total_deployed, price_p_sol_per_ore, motherlode_now, ore_haircut,
                                    )
                                );
                                if f_after > f_before {
                                    set = trial;
                                }
                            }
                        }
                    }
                    if set.is_empty() {
                        if debug {
                            // Optional diagnostic: show best unit EV and EV for top-1 candidate as a set
                            if let Some((best_idx, best_ev_unit)) = unit_ev_pairs.first().copied() {
                                let ev1_lamports = ev_set_total_sol_lamports(
                                    common_lamports, &[best_idx], &round, total_deployed, price_p_sol_per_ore, motherlode_now, ore_haircut
                                );
                                println!(
                                    "R={} no profitable set after interaction; best_unit_ev={:.3e} price={:.6} est_set_ev1≈{:.6} SOL (square={})",
                                    board.round_id,
                                    best_ev_unit,
                                    price_p_sol_per_ore,
                                    ev1_lamports / LAMPORTS_PER_SOL as f64,
                                    best_idx
                                );
                            } else {
                                println!(
                                    "R={} no profitable set after interaction; no candidates; price={:.6}",
                                    board.round_id,
                                    price_p_sol_per_ore
                                );
                            }
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                        continue;
                    }
                    // Kelly-like budget scaling (SOL-only) based on provisional a=common_lamports
                    // Compute outcomes and Kelly fraction
                    let outcomes = set_outcomes_total_sol_lamports(
                        common_lamports,
                        &set,
                        &round,
                        total_deployed,
                        price_p_sol_per_ore,
                        motherlode_now,
                        ore_haircut,
                    );
                    let f_kelly = kelly_fraction_from_outcomes(&outcomes);
                    // Fractional Kelly with clamps (reuse existing bounds)
                    let alpha = 0.5_f64;
                    let kelly_budget_pct = (alpha * f_kelly).clamp(min_budget_pct, max_budget_pct);
                    let kelly_budget_lamports = ((sol_balance as f64) * kelly_budget_pct) as u64;
                    // Apply conservative cap vs previously computed deploy_budget
                    if kelly_budget_lamports < deploy_budget {
                        if debug {
                            println!(
                                "R={} kelly: f={:.6} alpha={} pct={:.3}% old_budg={} new_budg={}",
                                board.round_id,
                                f_kelly,
                                alpha,
                                kelly_budget_pct * 100.0,
                                fmt_sol(deploy_budget),
                                fmt_sol(kelly_budget_lamports),
                            );
                        }
                        deploy_budget = kelly_budget_lamports;
                    }
                    // Recompute per-square cap for final set size and clamp amount if needed
                    let count_set = set.len().max(1);
                    let per_square_cap2 = (deploy_budget / (count_set as u64)).max(min_per_square_lamports) as f64;
                    let mut per_square_limits_set: Vec<f64> = Vec::new();
                    for &i in set.iter() {
                        let d_i = round.deployed[i] as f64;
                        let limit = max_profitable_amount(try_amount, per_square_cap2, d_i, total_deployed as f64, motherlode_now, price_p_sol_per_ore, ore_haircut);
                        if limit >= try_amount { per_square_limits_set.push(limit); }
                    }
                    if per_square_limits_set.len() < count_set {
                        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                        continue;
                    }
                    let common_lamports = common_lamports.min(per_square_limits_set.into_iter().fold(per_square_cap2, |m, x| m.min(x)).floor() as u64);

                    // Debug pre-deploy line
                    if debug {
                        let slots_left_dbg = board.end_slot.saturating_sub(current_slot);
                        let total_lamports = common_lamports.saturating_mul(set.len() as u64);
                        println!(
                            "R={} slots_left={} sol={} buf={} budg={} (pct={:.3}%) price={:.6} ORE motherlode={:.3} cand={} sel={} per={} total={} ev_avg={:.3e}",
                            board.round_id,
                            slots_left_dbg,
                            fmt_sol(sol_balance),
                            fmt_sol(buffer_target_lamports),
                            fmt_sol(deploy_budget),
                            budget_pct * 100.0,
                            price_p_sol_per_ore,
                            (treasury.motherlode as f64) / ore_api::consts::ONE_ORE as f64,
                            selected,
                            set.len(),
                            fmt_sol(common_lamports),
                            fmt_sol(total_lamports),
                            avg_edge_per_lamport,
                        );
                    }

                    // Before sending, re-check current board/round to avoid race at rollover
                    let board_re = match get_board(rpc).await { Ok(b) => b, Err(_) => board };
                    let mut current_slot_re = *slot_rx.borrow();
                    if current_slot_re == 0 {
                        current_slot_re = match get_current_slot(rpc).await { Ok(s) => s, Err(_) => current_slot };
                    }
                    if board_re.round_id != board.round_id {
                        if debug {
                            println!("R={} skipped deploy: round rolled to {}", board.round_id, board_re.round_id);
                        }
                    } else if !(current_slot_re >= board_re.start_slot && current_slot_re < board_re.end_slot) {
                        if debug {
                            println!(
                                "R={} skipped deploy: outside window (slot={}, start={}, end={})",
                                board_re.round_id,
                                current_slot_re,
                                board_re.start_slot,
                                board_re.end_slot
                            );
                        }
                    } else {
                        // Refresh price once more just before send
                        price_p_sol_per_ore = get_price_cached(&jupiter, &mut price_cache, 5_000).await;
                        // Build deploy ix with fixed entropy var address expected by program
                        let mut mask = [false; 25];
                        for &i in set.iter() { mask[i] = true; }
                        let count = set.len();
                        let deploy_ix = deploy_with_fixed_var(
                            payer.pubkey(),
                            payer.pubkey(),
                            common_lamports,
                            board.round_id,
                            mask,
                        );

                        // Single-tx send: bundle checkpoint (if required) + deploy. We closed any existing Automation at startup.
                        let mut ixs = Vec::new();
                        // Re-check miner right before send; if previous round is pending, bundle checkpoint first
                        if let Ok(m2) = get_miner(rpc, payer.pubkey()).await {
                            let prev_round_pending = m2.checkpoint_id < m2.round_id && m2.round_id < board.round_id;
                            if prev_round_pending {
                                ixs.push(ore_api::sdk::checkpoint(payer.pubkey(), payer.pubkey(), m2.round_id));
                            }
                        }
                        ixs.push(deploy_ix.clone());

                        if debug {
                            let slots_left_dbg = board_re.end_slot.saturating_sub(current_slot_re);
                            println!(
                                "window round={} slot={} start={} end={} slots_left={} close_auto_at_startup=true",
                                board.round_id,
                                current_slot_re,
                                board_re.start_slot,
                                board_re.end_slot,
                                slots_left_dbg,
                            );
                            simulate_with_budget(rpc, payer, &ixs).await;
                        }

                        // Compute and print estimated returns for this deployment (verbose only)
                        let (est_ev_sol, est_ev_ore, est_ev_total_sol, est_squares_csv) = if verbose {
                            let a = common_lamports as f64;
                            let total = total_deployed as f64;
            let mut ev_sol_sum = 0.0_f64;
            let mut ev_ore_sum = 0.0_f64;
            for &i in set.iter() {
                let d_i = round.deployed[i] as f64;
                ev_sol_sum += ev_sol_only(a, d_i, total);
                ev_ore_sum += ev_ore_component(a, d_i, motherlode_now);
            }
            let est_ev_sol = ev_sol_sum / LAMPORTS_PER_SOL as f64;
            let est_ev_ore = ev_ore_sum;
            let est_ev_total_sol = est_ev_sol
                + if price_p_sol_per_ore > 0.0 { ore_haircut * price_p_sol_per_ore * est_ev_ore } else { 0.0 };
                            let squares_csv = {
                                let mut v: Vec<String> = Vec::new();
                                for i in 0..25 { if mask[i] { v.push(i.to_string()); } }
                                v.join(",")
                            };
                            println!(
                                "R={} EST ev_sol={:.6} ev_ore={:.6} ev_total_sol≈{:.6} squares={}",
                                board.round_id, est_ev_sol, est_ev_ore, est_ev_total_sol, squares_csv
                            );
                            (est_ev_sol, est_ev_ore, est_ev_total_sol, squares_csv)
                        } else { (0.0, 0.0, 0.0, String::new()) };

                        // Summary line (optional): compute mu/sigma from outcomes and print one-line status
                        if summary {
                            let outcomes_sum = set_outcomes_total_sol_lamports(
                                common_lamports,
                                &set,
                                &round,
                                total_deployed,
                                price_p_sol_per_ore,
                                motherlode_now,
                                ore_haircut,
                            );
                            let p = 1.0_f64 / 25.0_f64;
                            let mu_lamports = outcomes_sum.iter().copied().sum::<f64>() * p;
                            let var_lamports2 = outcomes_sum.iter().map(|x| {
                                let d = *x - mu_lamports;
                                d * d
                            }).sum::<f64>() * p;
                            let sigma_lamports = var_lamports2.sqrt();
                            let (n1, n2) = if mu_lamports > 0.0 {
                                let r = sigma_lamports / mu_lamports;
                                ((r * r).round() as u64, ((2.0 * r) * (2.0 * r)).round() as u64)
                            } else { (0, 0) };
                            let seen = start_round_id.map(|s| board.round_id.saturating_sub(s)).unwrap_or(0);
                            let curr_sol = sol_balance;
                            let pnl_lamports = curr_sol.saturating_sub(start_sol_balance);
                            println!(
                                "SUMMARY R={} seen={} entered={} start={} curr={} pnl={} k={} per={} EV={:.6} mu={:.6} sigma={:.6} N1≈{} N2≈{} price={:.6} hc={:.2}",
                                board.round_id,
                                seen,
                                rounds_entered,
                                fmt_sol(start_sol_balance),
                                fmt_sol(curr_sol),
                                fmt_sol(pnl_lamports),
                                set.len(),
                                fmt_sol(common_lamports),
                                est_ev_total_sol,
                                mu_lamports / LAMPORTS_PER_SOL as f64,
                                sigma_lamports / LAMPORTS_PER_SOL as f64,
                                n1,
                                n2,
                                price_p_sol_per_ore,
                                ore_haircut,
                            );
                        }

                        let ok_to_send = match simulate_for_ok_cached(rpc, payer, &ixs, &bh_cache, Some(current_slot_re)).await {
                            Ok(true) => true,
                            Ok(false) => {
                                if verbose { println!("R={} skip send: simulate failed", board.round_id); }
                                false
                            }
                            Err(e) => {
                                if debug { println!("simulate error: {e:#?}"); }
                                false
                            }
                        };

                        if ok_to_send && !dry_run {
                            match submit_transaction_ws_confirm_cached(rpc, &ws, payer, &ixs, &bh_cache, Some(current_slot_re)).await {
                                Ok(sig) => {
                                    if verbose {
                                        println!("R={} deploy tx={}", board.round_id, sig);
                                    }
                                    rounds_entered = rounds_entered.saturating_add(1);
                                    last_played = Some((board.round_id, mask, common_lamports, count));
                                    if verbose {
                                        // Print submission + estimated returns summary
                                        println!(
                                            "R={} submitted per={} sel={} squares={} EST ev_sol={:.6} ev_ore={:.6} ev_total_sol≈{:.6}",
                                            board.round_id,
                                            fmt_sol(common_lamports),
                                            count,
                                            est_squares_csv,
                                            est_ev_sol,
                                            est_ev_ore,
                                            est_ev_total_sol,
                                        );
                                    }
                                }
                                Err(e) => {
                                    println!("deploy submit error: {e:#?}");
                                }
                            }
                        } else if verbose && dry_run {
                            let squares_csv = {
                                let mut v: Vec<String> = Vec::new();
                                for i in 0..25 { if mask[i] { v.push(i.to_string()); } }
                                v.join(",")
                            };
                            println!(
                                "R={} DRY-RUN deploy per={} sel={} squares={} total={}",
                                board.round_id,
                                fmt_sol(common_lamports),
                                count,
                                squares_csv,
                                fmt_sol(common_lamports.saturating_mul(count as u64))
                            );
                        }
                    }
                } else {
                    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                }
            }
        }

        // Wait for round increment (reset) using WS updates when available
        let old_round_id = board.round_id;
        loop {
            if let Some(b2) = board_rx.borrow().clone() {
                if b2.round_id > old_round_id { break; }
            }
            if board_rx.changed().await.is_err() { break; }
        }
        logged_predeploy_checked_round = None;

        // Determine if we won the previous round (used to decide checkpoint/claim)
        let mut won_prev_round = false;
        if let Some((rid, mask_played, _per_played, _sel_played)) = last_played {
            if rid == old_round_id {
                if let Ok(prev_round_state) = get_round(rpc, rid).await {
                    if let Some(rng) = prev_round_state.rng() {
                        let win_square = prev_round_state.winning_square(rng);
                        won_prev_round = mask_played[win_square];
                    }
                }
            }
        }

        // If we played this round, print result (win square and whether we won)
        if verbose {
            if let Some((rid, mask_played, per_played, sel_played)) = last_played {
                if rid == old_round_id {
                    if let Ok(prev_round) = get_round(rpc, rid).await {
                        if let Some(rng) = prev_round.rng() {
                            let win_square = prev_round.winning_square(rng);
                            let won = mask_played[win_square];
                            println!(
                                "R={} result win_square={} won={} per={} sel={}",
                                rid,
                                win_square,
                                won,
                                fmt_sol(per_played),
                                sel_played,
                            );
                        } else {
                            println!("R={} result unavailable (rng not finalized)", rid);
                        }
                    }
                }
            }
        }

        // Checkpoint (only if needed) and claim SOL (only if >0)
        if !dry_run && won_prev_round {
            // Load miner state
            let miner_before = get_miner(rpc, payer.pubkey()).await.ok();
            let need_checkpoint = miner_before
                .as_ref()
                .map(|m| m.checkpoint_id < old_round_id && m.round_id == old_round_id)
                .unwrap_or(true);
            if need_checkpoint {
                let _ = submit_transaction(
                    rpc,
                    payer,
                    &[ore_api::sdk::checkpoint(payer.pubkey(), payer.pubkey(), old_round_id)],
                )
                .await;
                // After checkpoint, print rewards available (SOL and ORE)
                if verbose {
                    if let Ok(miner_chk) = get_miner(rpc, payer.pubkey()).await {
                        println!(
                            "R={} rewards available: sol={} ore={}",
                            old_round_id,
                            fmt_sol(miner_chk.rewards_sol),
                            fmt_ore(miner_chk.rewards_ore),
                        );
                    }
                }
            } else if debug {
                println!("R={} skip checkpoint (already checkpointed)", old_round_id);
            }

            // Re-read miner to check rewards
            if let Ok(miner_after) = get_miner(rpc, payer.pubkey()).await {
                if miner_after.rewards_sol > 0 {
                    let _ = submit_transaction(rpc, payer, &[ore_api::sdk::claim_sol(payer.pubkey())]).await;
                }
            }
        } else if debug && !won_prev_round {
            println!("R={} skip checkpoint (lost round)", old_round_id);
        }

        // If SOL below buffer, optionally claim ORE and swap to SOL (only if we won and have ORE)
        if !dry_run && won_prev_round {
            let sol_after = rpc.get_balance(&payer.pubkey()).await.unwrap_or(0);
            if debug { println!("R={} post-claim sol={}", old_round_id, fmt_sol(sol_after)); }
            if sol_after < buffer_target_lamports {
                if let Ok(miner_now) = get_miner(rpc, payer.pubkey()).await {
                    if miner_now.rewards_ore > 0 {
                        // Claim ORE
                        let _ = submit_transaction(rpc, payer, &[ore_api::sdk::claim_ore(payer.pubkey())]).await;
                        // Compute missing SOL and swap required ORE
                        if price_p_sol_per_ore > 0.0 {
                            let needed_lamports = buffer_target_lamports.saturating_sub(sol_after);
                            let needed_ore = (needed_lamports as f64 / price_p_sol_per_ore) / (LAMPORTS_PER_SOL as f64);
							let needed_ore_grams = (needed_ore * ONE_ORE as f64) as u64;
							// Clamp by available ORE ATA balance
							let ore_ata = get_associated_token_address(&payer.pubkey(), &ore_api::consts::MINT_ADDRESS);
							let available_ore_grams = rpc
								.get_token_account_balance(&ore_ata)
								.await
								.ok()
								.and_then(|ui| ui.amount.parse::<u64>().ok())
								.unwrap_or(0);
							let swap_grams = needed_ore_grams.min(available_ore_grams);
							if swap_grams > 0 {
								if debug {
									println!(
										"R={} topping up: need_sol={} ore_needed={} ore_avail={} ore_swapping={}",
										old_round_id,
										fmt_sol(needed_lamports),
										fmt_ore(needed_ore_grams),
										fmt_ore(available_ore_grams),
										fmt_ore(swap_grams)
									);
								}
								let _ = swap_ore_to_sol(rpc, payer, &jupiter, swap_grams).await;
							} else if debug {
								println!("R={} skip top-up: no ORE available to swap", old_round_id);
							}
                        }
                    } else if debug {
                        println!("R={} skip top-up: no ORE rewards", old_round_id);
                    }
                }
            }
        } else if debug && !won_prev_round {
            println!("R={} skip top-up: lost round", old_round_id);
        }
    }
}

// Backtest command: simulate policy over historical rounds without submitting transactions.
async fn backtest(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
)-> Result<(), anyhow::Error> {
    // Config (env or defaults) - reuse knobs
    let price_api_base = std::env::var("API_BASE_URL").unwrap_or("https://lite-api.jup.ag/swap/v1".into());
    let jupiter = JupiterSwapApiClient::new(price_api_base);
    let min_buffer_sol: f64 = std::env::var("AUTO_MIN_SOL_BUFFER_SOL").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);
    let min_budget_pct: f64 = std::env::var("AUTO_MIN_BUDGET_PCT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.005);
    let max_budget_pct: f64 = std::env::var("AUTO_MAX_BUDGET_PCT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.02);
    let edge_scale: f64 = std::env::var("AUTO_EDGE_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let verbose: bool = std::env::var("AUTO_VERBOSE").ok().map(|v| v != "0").unwrap_or(false);
    let debug: bool = std::env::var("AUTO_DEBUG").ok().map(|v| v != "0").unwrap_or(false);
    let summary: bool = std::env::var("AUTO_SUMMARY").ok().map(|v| v != "0").unwrap_or(true);
    let max_squares: usize = std::env::var("AUTO_MAX_SQUARES").ok().and_then(|v| v.parse().ok()).unwrap_or(25);
    let min_per_square_lamports: u64 = std::env::var("AUTO_MIN_PER_SQUARE_LAMPORTS").ok().and_then(|v| v.parse().ok()).unwrap_or(10_000);
    let ignore_motherlode: bool = std::env::var("AUTO_IGNORE_MOTHERLODE").ok().map(|v| v != "0").unwrap_or(false);
    let ore_haircut: f64 = std::env::var("AUTO_ORE_HAIRCUT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.9);
    let fixed_price: Option<f64> = std::env::var("BACKTEST_FIXED_PRICE").ok().and_then(|v| v.parse().ok());
    let start_id: u64 = std::env::var("BACKTEST_START_ID").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        // Default start: current board - 1000 (approx)
        futures::executor::block_on(async {
            get_board(rpc).await.map(|b| b.round_id.saturating_sub(1000)).unwrap_or(1)
        })
    });
    let count: u64 = std::env::var("BACKTEST_COUNT").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let mc: bool = std::env::var("BACKTEST_MONTE_CARLO").ok().map(|v| v != "0").unwrap_or(false);
    let start_sol_balance_env: Option<f64> = std::env::var("BACKTEST_START_SOL").ok().and_then(|v| v.parse().ok());

    // Simulated bankroll
    let wallet_balance = rpc.get_balance(&payer.pubkey()).await.unwrap_or(0);
    let start_sol_balance_lamports = start_sol_balance_env
        .map(|s| (s * LAMPORTS_PER_SOL as f64) as u64)
        .unwrap_or(wallet_balance);
    let mut sim_balance = start_sol_balance_lamports;
    let buffer_target_lamports = (min_buffer_sol * LAMPORTS_PER_SOL as f64) as u64;

    // Price cache
    let mut price_cache: Option<(f64, Instant)> = None;

    // If no fixed price provided, fetch once up-front; print and abort if unavailable.
    if fixed_price.is_none() {
        let p = get_price_cached(&jupiter, &mut price_cache, 5_000).await;
        if p <= 0.0 {
            println!("BACKTEST abort: failed to fetch live ORE→SOL price (got 0). Set BACKTEST_FIXED_PRICE or check API_BASE_URL.");
            return Err(anyhow::anyhow!("price unavailable"));
        } else {
            println!("BACKTEST using live price (SOL/ORE) = {:.6}", p);
        }
    }

    let mut rounds_entered: u64 = 0;
    let mut seen: u64 = 0;

    for i in 0..count {
        let rid = start_id + i;
        seen += 1;

        let round = match get_round(rpc, rid).await {
            Ok(r) => r,
            Err(_) => continue,
        };

        // Skip rounds with no rng (treat as refund): we won't deploy
        let rng_opt = round.rng();

        // Use round.motherlode as snapshot proxy
        let motherlode_now = if ignore_motherlode { 0.0 } else { round.motherlode as f64 / ONE_ORE as f64 };

        // Price (fixed or cached)
        let mut price_p_sol_per_ore = fixed_price.unwrap_or_else(|| 0.0);
        if price_p_sol_per_ore == 0.0 {
            price_p_sol_per_ore = get_price_cached(&jupiter, &mut price_cache, 5_000).await;
        }

        // Build EV at unit amount for candidates
        let total_deployed: u64 = round.deployed.iter().sum();
        let mut unit_ev_pairs: Vec<(usize, f64)> = Vec::new();
        let try_amount = min_per_square_lamports as f64;
        for i_sq in 0..25 {
            let d_i = round.deployed[i_sq] as f64;
            let total = total_deployed as f64;
            let ev_sol = ev_sol_only(try_amount, d_i, total);
            let ev_ore = ev_ore_component(try_amount, d_i, motherlode_now);
            let ev_total = if price_p_sol_per_ore > 0.0 { ev_sol + ore_haircut * price_p_sol_per_ore * ev_ore } else { ev_sol };
            if ev_total > 0.0 { unit_ev_pairs.push((i_sq, ev_total)); }
        }
        unit_ev_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if unit_ev_pairs.is_empty() {
            if summary {
                println!(
                    "SUMMARY R={} seen={} entered={} start={} curr={} pnl={} k=0 per=0 EV=0.000000 mu=0.000000 sigma=0.000000 N1≈0 N2≈0 price={:.6} hc={:.2}",
                    rid,
                    seen - 1,
                    rounds_entered,
                    fmt_sol(start_sol_balance_lamports),
                    fmt_sol(sim_balance),
                    fmt_sol(sim_balance.saturating_sub(start_sol_balance_lamports)),
                    price_p_sol_per_ore,
                    ore_haircut,
                );
            }
            continue;
        }

        // Heuristic budget by edge
        let selected = std::cmp::min(max_squares, unit_ev_pairs.len());
        let candidates: Vec<usize> = unit_ev_pairs.iter().take(selected).map(|(i, _)| *i).collect();
        let avg_edge_per_lamport = unit_ev_pairs.iter().take(selected)
            .map(|(_, ev)| ev / try_amount)
            .sum::<f64>() / (selected as f64);
        let budget_pct = (avg_edge_per_lamport * edge_scale).clamp(min_budget_pct, max_budget_pct);
        let spendable = sim_balance.saturating_sub(buffer_target_lamports);
        if spendable < min_per_square_lamports {
            if summary {
                println!(
                    "SUMMARY R={} seen={} entered={} start={} curr={} pnl={} k=0 per=0 EV=0.000000 mu=0.000000 sigma=0.000000 N1≈0 N2≈0 price={:.6} hc={:.2}",
                    rid, seen - 1, rounds_entered, fmt_sol(start_sol_balance_lamports), fmt_sol(sim_balance),
                    fmt_sol(sim_balance.saturating_sub(start_sol_balance_lamports)),
                    price_p_sol_per_ore, ore_haircut
                );
            }
            continue;
        }
        let budget_lamports = ((sim_balance as f64) * budget_pct) as u64;
        let mut deploy_budget = spendable.min(budget_lamports);

        // Initial mask/count/per cap
        let mut mask = [false; 25];
        for i_sq in candidates.iter().copied() { mask[i_sq] = true; }
        let count = candidates.len().max(1);
        let per_square_cap = (deploy_budget / (count as u64)).max(min_per_square_lamports) as f64;

        // Per-square profitable limits and common amount
        let mut per_square_limits: Vec<f64> = Vec::new();
        for &i_sq in candidates.iter() {
            let d_i = round.deployed[i_sq] as f64;
            let total = total_deployed as f64;
            let limit = max_profitable_amount(try_amount, per_square_cap, d_i, total, motherlode_now, price_p_sol_per_ore, ore_haircut);
            if limit >= try_amount { per_square_limits.push(limit); }
        }
        if per_square_limits.len() < count {
            if summary {
                println!(
                    "SUMMARY R={} seen={} entered={} start={} curr={} pnl={} k=0 per=0 EV=0.000000 mu=0.000000 sigma=0.000000 N1≈0 N2≈0 price={:.6} hc={:.2}",
                    rid, seen - 1, rounds_entered, fmt_sol(start_sol_balance_lamports), fmt_sol(sim_balance),
                    fmt_sol(sim_balance.saturating_sub(start_sol_balance_lamports)),
                    price_p_sol_per_ore, ore_haircut
                );
            }
            continue;
        }
        let common_amount = per_square_limits.into_iter().fold(per_square_cap, |m, x| m.min(x));
        let mut common_lamports = common_amount.floor() as u64;
        if common_lamports < min_per_square_lamports {
            if summary {
                println!(
                    "SUMMARY R={} seen={} entered={} start={} curr={} pnl={} k=0 per=0 EV=0.000000 mu=0.000000 sigma=0.000000 N1≈0 N2≈0 price={:.6} hc={:.2}",
                    rid, seen - 1, rounds_entered, fmt_sol(start_sol_balance_lamports), fmt_sol(sim_balance),
                    fmt_sol(sim_balance.saturating_sub(start_sol_balance_lamports)),
                    price_p_sol_per_ore, ore_haircut
                );
            }
            continue;
        }

        // Interaction-aware set selection with Kelly tiebreak
        let ordered = candidates.clone();
        let mut set: Vec<usize> = Vec::new();
        for &i_sq in ordered.iter() {
            if set.len() >= max_squares { break; }
            let mut trial = set.clone();
            trial.push(i_sq);
            let ev_before = ev_set_total_sol_lamports(common_lamports, &set, &round, total_deployed, price_p_sol_per_ore, motherlode_now, ore_haircut);
            let ev_after  = ev_set_total_sol_lamports(common_lamports, &trial, &round, total_deployed, price_p_sol_per_ore, motherlode_now, ore_haircut);
            if ev_after > ev_before {
                set = trial;
            } else {
                let eps_lamports = (1e-6f64) * (LAMPORTS_PER_SOL as f64);
                if ev_after >= ev_before - eps_lamports {
                    let f_before = kelly_fraction_from_outcomes(&set_outcomes_total_sol_lamports(
                        common_lamports, &set, &round, total_deployed, price_p_sol_per_ore, motherlode_now, ore_haircut
                    ));
                    let f_after = kelly_fraction_from_outcomes(&set_outcomes_total_sol_lamports(
                        common_lamports, &trial, &round, total_deployed, price_p_sol_per_ore, motherlode_now, ore_haircut
                    ));
                    if f_after > f_before { set = trial; }
                }
            }
        }
        if set.is_empty() {
            if summary {
                println!(
                    "SUMMARY R={} seen={} entered={} start={} curr={} pnl={} k=0 per=0 EV=0.000000 mu=0.000000 sigma=0.000000 N1≈0 N2≈0 price={:.6} hc={:.2}",
                    rid, seen - 1, rounds_entered, fmt_sol(start_sol_balance_lamports), fmt_sol(sim_balance),
                    fmt_sol(sim_balance.saturating_sub(start_sol_balance_lamports)),
                    price_p_sol_per_ore, ore_haircut
                );
            }
            continue;
        }

        // Kelly budget adjustment
        let outcomes = set_outcomes_total_sol_lamports(common_lamports, &set, &round, total_deployed, price_p_sol_per_ore, motherlode_now, ore_haircut);
        let p_out = 1.0_f64 / 25.0_f64;
        let mu_lamports = outcomes.iter().copied().sum::<f64>() * p_out;
        let var_lamports2 = outcomes.iter().map(|x| { let d = *x - mu_lamports; d*d }).sum::<f64>() * p_out;
        let sigma_lamports = var_lamports2.sqrt();
        let f_kelly = if var_lamports2 > 0.0 { (mu_lamports / var_lamports2).max(0.0) } else { 0.0 };
        let alpha = 0.5_f64;
        let kelly_budget_pct = (alpha * f_kelly).clamp(min_budget_pct, max_budget_pct);
        let kelly_budget_lamports = ((sim_balance as f64) * kelly_budget_pct) as u64;
        if kelly_budget_lamports < deploy_budget {
            deploy_budget = kelly_budget_lamports;
        }

        // Recompute per-square caps for final set size
        let count_set = set.len().max(1);
        let per_square_cap2 = (deploy_budget / (count_set as u64)).max(min_per_square_lamports) as f64;
        let mut per_square_limits_set: Vec<f64> = Vec::new();
        for &i_sq in set.iter() {
            let d_i = round.deployed[i_sq] as f64;
            let limit = max_profitable_amount(try_amount, per_square_cap2, d_i, total_deployed as f64, motherlode_now, price_p_sol_per_ore, ore_haircut);
            if limit >= try_amount { per_square_limits_set.push(limit); }
        }
        if per_square_limits_set.len() < count_set {
            if summary {
                println!(
                    "SUMMARY R={} seen={} entered={} start={} curr={} pnl={} k=0 per=0 EV=0.000000 mu=0.000000 sigma=0.000000 N1≈0 N2≈0 price={:.6} hc={:.2}",
                    rid, seen - 1, rounds_entered, fmt_sol(start_sol_balance_lamports), fmt_sol(sim_balance),
                    fmt_sol(sim_balance.saturating_sub(start_sol_balance_lamports)),
                    price_p_sol_per_ore, ore_haircut
                );
            }
            continue;
        }
        common_lamports = common_lamports.min(per_square_limits_set.into_iter().fold(per_square_cap2, |m, x| m.min(x)).floor() as u64);
        if common_lamports < min_per_square_lamports {
            if summary {
                println!(
                    "SUMMARY R={} seen={} entered={} start={} curr={} pnl={} k=0 per=0 EV=0.000000 mu=0.000000 sigma=0.000000 N1≈0 N2≈0 price={:.6} hc={:.2}",
                    rid, seen - 1, rounds_entered, fmt_sol(start_sol_balance_lamports), fmt_sol(sim_balance),
                    fmt_sol(sim_balance.saturating_sub(start_sol_balance_lamports)),
                    price_p_sol_per_ore, ore_haircut
                );
            }
            continue;
        }

        // Compute EST EV total (SOL)
        let a = common_lamports as f64;
        let total = total_deployed as f64;
        let mut ev_sol_sum = 0.0_f64;
        let mut ev_ore_sum = 0.0_f64;
        let mut mask_set = [false; 25];
        for &i_sq in set.iter() {
            mask_set[i_sq] = true;
            let d_i = round.deployed[i_sq] as f64;
            ev_sol_sum += ev_sol_only(a, d_i, total);
            ev_ore_sum += ev_ore_component(a, d_i, motherlode_now);
        }
        let est_ev_sol = ev_sol_sum / LAMPORTS_PER_SOL as f64;
        let est_ev_ore = ev_ore_sum;
        let est_ev_total_sol = est_ev_sol + if price_p_sol_per_ore > 0.0 { ore_haircut * price_p_sol_per_ore * est_ev_ore } else { 0.0 };

        // mu/sigma and N1/N2
        let (n1, n2) = if mu_lamports > 0.0 {
            let r = sigma_lamports / mu_lamports;
            ((r * r).round() as u64, ((2.0 * r) * (2.0 * r)).round() as u64)
        } else { (0, 0) };

        // Realized outcome for this round (deterministic; MC optional for non-split ORE top miner)
        let sol_pnl_lamports: i128;
        let mut ore_grams: f64 = 0.0;
        if let Some(rng) = rng_opt {
            let win = round.winning_square(rng);
            let k = set.len() as u64;
            // Deposit paid
            if mask_set[win] {
                let d_win = round.deployed[win] + common_lamports;
                let total_with_us = total_deployed + k * common_lamports;
                let losers_pool = (total_with_us.saturating_sub(d_win)) as f64;
                let d_win_f = d_win as f64;
                let share = a / d_win_f;
                let win_gain = 0.99_f64 * a + 0.891_f64 * losers_pool * share;
                let other_losses = a * ((k - 1) as f64);
                sol_pnl_lamports = (win_gain - other_losses) as i128;
                // Motherlode realized (if any)
                if round.motherlode > 0 {
                    ore_grams += (round.motherlode as f64) * share;
                }
                // Top miner ORE
                if round.is_split_reward(rng) {
                    ore_grams += ONE_ORE as f64 * share;
                } else {
                    if mc {
                        // MC draw from rng for determinism
                        let u = (rng.reverse_bits() as f64) / (u64::MAX as f64);
                        if u < share { ore_grams += ONE_ORE as f64; }
                    } else {
                        // Deterministic EV
                        ore_grams += ONE_ORE as f64 * share;
                    }
                }
            } else {
                sol_pnl_lamports = -((k * common_lamports) as i128);
            }
        } else {
            // No RNG ⇒ refund scenario; no net effect
            sol_pnl_lamports = 0;
        }
        let ore_value_lamports = if price_p_sol_per_ore > 0.0 {
            (ore_haircut * price_p_sol_per_ore * (ore_grams / ONE_ORE as f64) * LAMPORTS_PER_SOL as f64) as i128
        } else { 0 };
        let total_pnl_lamports = sol_pnl_lamports + ore_value_lamports;
        sim_balance = ((sim_balance as i128) + total_pnl_lamports).max(0) as u64;
        rounds_entered = rounds_entered.saturating_add(1);

        if summary {
            println!(
                "SUMMARY R={} seen={} entered={} start={} curr={} pnl={} k={} per={} EV={:.6} mu={:.6} sigma={:.6} N1≈{} N2≈{} price={:.6} hc={:.2}",
                rid,
                seen - 1,
                rounds_entered,
                fmt_sol(start_sol_balance_lamports),
                fmt_sol(sim_balance),
                fmt_sol(sim_balance.saturating_sub(start_sol_balance_lamports)),
                set.len(),
                fmt_sol(common_lamports),
                est_ev_total_sol,
                mu_lamports / LAMPORTS_PER_SOL as f64,
                sigma_lamports / LAMPORTS_PER_SOL as f64,
                n1,
                n2,
                price_p_sol_per_ore,
                ore_haircut,
            );
        } else if verbose {
            let squares_csv = {
                let mut v: Vec<String> = Vec::new();
                for i_sq in 0..25 { if mask_set[i_sq] { v.push(i_sq.to_string()); } }
                v.join(",")
            };
            println!(
                "BT R={} sel={} per={} squares={} est_ev_total={:.6} realized_pnl={}",
                rid,
                set.len(),
                fmt_sol(common_lamports),
                squares_csv,
                est_ev_total_sol,
                fmt_sol(total_pnl_lamports.max(0) as u64),
            );
        }
    }

    // Final line
    println!(
        "BACKTEST done: start={} curr={} pnl={} seen={} entered={}",
        fmt_sol(start_sol_balance_lamports),
        fmt_sol(sim_balance),
        fmt_sol(sim_balance.saturating_sub(start_sol_balance_lamports)),
        seen.saturating_sub(1),
        rounds_entered
    );

    Ok(())
}

fn ev_sol_only(a: f64, d_i: f64, total: f64) -> f64 {
    let p_win = 1.0 / 25.0;
    let p_lose = 1.0 - p_win;
    let d_win = d_i + a;
    let losers_pool = (total - d_i).max(0.0);
    let share_from_pool = 0.891 * losers_pool * (a / d_win);
    p_win * (0.99 * a + share_from_pool) - p_lose * a
}

fn ev_ore_component(a: f64, d_i: f64, motherlode_now_ore: f64) -> f64 {
    let p_win = 1.0 / 25.0;
    let d_win = d_i + a;
    let ore_per_win = a / d_win;
    // Only the existing motherlode contributes EV this round; the +0.2 ORE mint applies to the next round.
    let motherlode_ev = motherlode_now_ore / 625.0;
    p_win * ore_per_win * (1.0 + motherlode_ev)
}

fn ev_total(a: f64, d_i: f64, total: f64, motherlode_now_ore: f64, price_p_sol_per_ore: f64, ore_haircut: f64) -> f64 {
    let sol = ev_sol_only(a, d_i, total);
    if price_p_sol_per_ore > 0.0 {
        // Convert ORE EV valued in SOL to lamports to match `ev_sol_only` units.
        sol + ore_haircut * price_p_sol_per_ore * (LAMPORTS_PER_SOL as f64) * ev_ore_component(a, d_i, motherlode_now_ore)
    } else {
        sol
    }
}

fn max_profitable_amount(
    a_min: f64,
    a_max: f64,
    d_i: f64,
    total: f64,
    motherlode_now_ore: f64,
    price_p_sol_per_ore: f64,
    ore_haircut: f64,
) -> f64 {
    if a_max <= a_min { return 0.0; }
    let ev_at_max = ev_total(a_max, d_i, total, motherlode_now_ore, price_p_sol_per_ore, ore_haircut);
    if ev_at_max > 0.0 { return a_max; }
    let ev_at_min = ev_total(a_min, d_i, total, motherlode_now_ore, price_p_sol_per_ore, ore_haircut);
    if ev_at_min <= 0.0 { return 0.0; }
    // Bisection to find root where EV -> 0
    let mut lo = a_min;
    let mut hi = a_max;
    for _ in 0..32 {
        let mid = 0.5 * (lo + hi);
        let ev_mid = ev_total(mid, d_i, total, motherlode_now_ore, price_p_sol_per_ore, ore_haircut);
        if ev_mid > 0.0 { lo = mid; } else { hi = mid; }
    }
    lo
}

// Interaction-aware set EV (SOL only), accounting for self-cannibalization across chosen squares.
fn ev_set_sol_lamports(
    a_lamports: u64,
    set: &[usize],
    round: &Round,
    total_deployed: u64,
)-> f64 {
    let a = a_lamports as f64;
    let k = set.len() as u64;
    if k == 0 { return 0.0; }
    let total_with_us = total_deployed as f64 + (k as f64) * a;
    let mut in_set = [false; 25];
    for &i in set { in_set[i] = true; }
    let mut ev_sum = 0.0;
    for w in 0..25 {
        if in_set[w] {
            let d_win = round.deployed[w] as f64 + a;
            let losers_pool = total_with_us - d_win;
            let win_gain = 0.99 * a + 0.891 * losers_pool * (a / d_win);
            let other_losses = a * ((k - 1) as f64);
            ev_sum += win_gain - other_losses;
        } else {
            ev_sum += -(a * (k as f64));
        }
    }
    ev_sum / 25.0
}

// Set-level EV including ORE valued in SOL lamports (haircut applied)
fn ev_set_total_sol_lamports(
    a_lamports: u64,
    set: &[usize],
    round: &Round,
    total_deployed: u64,
    price_p_sol_per_ore: f64,
    motherlode_now_ore: f64,
    ore_haircut: f64,
) -> f64 {
    let ev_sol = ev_set_sol_lamports(a_lamports, set, round, total_deployed);
    if price_p_sol_per_ore <= 0.0 || set.is_empty() { return ev_sol; }
    let a = a_lamports as f64;
    let ore_ev_sum = set.iter().map(|&i| {
        let d_win = round.deployed[i] as f64 + a;
        let ore_per_win = a / d_win;
        let motherlode_ev = motherlode_now_ore / 625.0;
        (1.0/25.0) * ore_per_win * (1.0 + motherlode_ev)
    }).sum::<f64>();
    let ore_ev_sol_lamports = ore_haircut * price_p_sol_per_ore * (LAMPORTS_PER_SOL as f64) * ore_ev_sum;
    ev_sol + ore_ev_sol_lamports
}

// Per-outcome returns (SOL lamports) for the selected set
fn set_outcomes_sol_lamports(
    a_lamports: u64,
    set: &[usize],
    round: &Round,
    total_deployed: u64,
) -> [f64; 25] {
    let a = a_lamports as f64;
    let k = set.len() as u64;
    let total_with_us = total_deployed as f64 + (k as f64) * a;
    let mut in_set = [false; 25];
    for &i in set { in_set[i] = true; }
    let mut out = [0.0_f64; 25];
    for w in 0..25 {
        if in_set[w] {
            let d_win = round.deployed[w] as f64 + a;
            let losers_pool = total_with_us - d_win;
            let win_gain = 0.99 * a + 0.891 * losers_pool * (a / d_win);
            let other_losses = a * ((k - 1) as f64);
            out[w] = win_gain - other_losses;
        } else {
            out[w] = -(a * (k as f64));
        }
    }
    out
}

// Per-outcome returns including ORE EV valued in SOL lamports (haircut on price)
fn set_outcomes_total_sol_lamports(
    a_lamports: u64,
    set: &[usize],
    round: &Round,
    total_deployed: u64,
    price_p_sol_per_ore: f64,
    motherlode_now_ore: f64,
    ore_haircut: f64,
) -> [f64; 25] {
    let mut base = set_outcomes_sol_lamports(a_lamports, set, round, total_deployed);
    if price_p_sol_per_ore <= 0.0 {
        return base;
    }
    let a = a_lamports as f64;
    let mut in_set = [false; 25];
    for &i in set { in_set[i] = true; }
    for w in 0..25 {
        if in_set[w] {
            let d_win = round.deployed[w] as f64 + a;
            let ore_per_win = a / d_win;
            let motherlode_ev = motherlode_now_ore / 625.0;
            let ore_on_outcome = ore_per_win * (1.0 + motherlode_ev);
            let ore_value_lamports = ore_haircut * price_p_sol_per_ore * (LAMPORTS_PER_SOL as f64) * ore_on_outcome;
            base[w] += ore_value_lamports;
        }
    }
    base
}

// Kelly fraction estimate: f ≈ μ / σ²
fn kelly_fraction_from_outcomes(outcomes: &[f64; 25]) -> f64 {
    let p = 1.0_f64 / 25.0_f64;
    let mu = outcomes.iter().copied().sum::<f64>() * p;
    let var = outcomes.iter().map(|x| {
        let d = *x - mu;
        d * d
    }).sum::<f64>() * p;
    if var <= 0.0 { return 0.0; }
    (mu / var).max(0.0)
}

fn fmt_sol(lamports: u64) -> String {
    format!("{:.6}", lamports_to_sol(lamports))
}

fn fmt_ore(grams: u64) -> String {
    format!(
        "{:.6}",
        (grams as f64) / ore_api::consts::ONE_ORE as f64
    )
}

// Convert HTTP RPC URL to WS URL if WS env not provided
fn http_to_ws(http: &str) -> String {
    if http.starts_with("https://") {
        format!("wss://{}", &http["https://".len()..])
    } else if http.starts_with("http://") {
        format!("ws://{}", &http["http://".len()..])
    } else {
        http.to_string()
    }
}

// (SlotRate removed; sniping now uses slots directly)

// Start a websocket slot subscription and publish latest slot via watch channel
async fn start_slot_watch(ws_url: String) -> Result<watch::Receiver<u64>, anyhow::Error> {
    let (tx, rx) = watch::channel(0u64);
    tokio::spawn(async move {
        loop {
            match PubsubClient::new(&ws_url).await {
                Ok(client) => {
                    match client.slot_subscribe().await {
                        Ok((mut stream, _unsubscribe)) => {
                            while let Some(update) = stream.next().await {
                                let SlotInfo { slot, .. } = update;
                                let _ = tx.send(slot);
                            }
                        }
                        Err(_) => {
                            tokio::time::sleep(Duration::from_millis(300)).await;
                        }
                    }
                }
                Err(_e) => {
                    // Backoff on reconnect
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        }
    });
    Ok(rx)
}

// Live current slot from the node (processed commitment)
async fn get_current_slot(rpc: &RpcClient) -> Result<u64, anyhow::Error> {
    Ok(rpc.get_slot_with_commitment(CommitmentConfig::processed()).await?)
}

//

fn deploy_with_fixed_var(
    signer: Pubkey,
    authority: Pubkey,
    amount: u64,
    round_id: u64,
    squares: [bool; 25],
) -> solana_sdk::instruction::Instruction {
    // Use SDK default account metas (matches official frontend behavior)
    ore_api::sdk::deploy(signer, authority, amount, round_id, squares)
}

// Simple tester: print time-left using WS-only slots and EWMA, monotonic decreasing display
async fn timeleft(rpc: &RpcClient) -> Result<(), anyhow::Error> {
    let ws = std::env::var("WS").unwrap_or_else(|_| http_to_ws(&std::env::var("RPC").expect("Missing RPC env var")));
    let slot_rx = start_slot_watch(ws).await?;
    let mut ticker = tokio::time::interval(Duration::from_secs(1));
    let max_secs: u64 = std::env::var("TIMELEFT_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(90);

    // Track round to reset display when round changes
    let mut last_round_id: Option<u64> = None;
    let start_t = Instant::now();

    loop {
        ticker.tick().await;

        // Fetch board fresh each second to capture rollover
        let board = match get_board(rpc).await {
            Ok(b) => b,
            Err(_) => continue,
        };

        if board.end_slot == u64::MAX {
            println!("R={} waiting for first deploy...", board.round_id);
            last_round_id = Some(board.round_id);
            continue;
        }

        // Current slot primarily from WS; fallback to RPC only if WS hasn’t delivered yet
        let mut slot_now = *slot_rx.borrow();
        if slot_now == 0 {
            slot_now = get_current_slot(rpc).await.unwrap_or(0);
            if slot_now == 0 {
                continue;
            }
        }

        // Reset on round change
        if last_round_id != Some(board.round_id) {
            last_round_id = Some(board.round_id);
        }

        let left_slots = board.end_slot.saturating_sub(slot_now);
        println!(
            "R={} slot={} end={} left_slots={}",
            board.round_id, slot_now, board.end_slot, left_slots
        );

        if start_t.elapsed().as_secs() >= max_secs {
            break;
        }
    }
    Ok(())
}

async fn get_ore_sol_price(jupiter: &JupiterSwapApiClient) -> Result<f64, anyhow::Error> {
    const INPUT_MINT: Pubkey = ore_api::consts::MINT_ADDRESS;
    const OUTPUT_MINT: Pubkey = ore_api::consts::SOL_MINT;
    // Quote 1 ORE
    let amount = ore_api::consts::ONE_ORE; // grams
    let quote_request = QuoteRequest { amount, input_mint: INPUT_MINT, output_mint: OUTPUT_MINT, max_accounts: Some(55), ..QuoteRequest::default() };
    let quote = jupiter.quote(&quote_request).await?;
    if quote.out_amount == 0 { return Ok(0.0); }
    // Price = SOL lamports per ORE
    let price_lamports_per_ore = quote.out_amount as f64;
    Ok(price_lamports_per_ore / LAMPORTS_PER_SOL as f64)
}

async fn get_price_cached(
    jupiter: &JupiterSwapApiClient,
    cache: &mut Option<(f64, Instant)>,
    ttl_ms: u64,
) -> f64 {
    let now = Instant::now();
    if let Some((p, ts)) = cache.as_ref() {
        if now.duration_since(*ts).as_millis() < ttl_ms as u128 {
            return *p;
        }
    }
    match get_ore_sol_price(jupiter).await {
        Ok(p) if p > 0.0 => { *cache = Some((p, now)); p }
        _ => {
            if let Some((p, _)) = cache.as_ref() { *p } else { 0.0 }
        }
    }
}

async fn swap_ore_to_sol(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
    jupiter: &JupiterSwapApiClient,
    in_amount_grams: u64,
) -> Result<(), anyhow::Error> {
    const INPUT_MINT: Pubkey = ore_api::consts::MINT_ADDRESS;
    const OUTPUT_MINT: Pubkey = ore_api::consts::SOL_MINT;
    let quote_request = QuoteRequest { amount: in_amount_grams, input_mint: INPUT_MINT, output_mint: OUTPUT_MINT, max_accounts: Some(55), ..QuoteRequest::default() };
    let quote_response = match jupiter.quote(&quote_request).await {
        Ok(q) => q,
        Err(e) => { println!("swap quote failed: {e:#?}"); return Err(anyhow::anyhow!("quote failed")); }
    };
    let out_amount_est = quote_response.out_amount;
    // Enable wrap/unwrap if either side is SOL
    let wrap_unwrap = INPUT_MINT == ore_api::consts::SOL_MINT || OUTPUT_MINT == ore_api::consts::SOL_MINT;
    let response = jupiter.swap_instructions(&SwapRequest {
        user_public_key: payer.pubkey(),
        quote_response: quote_response.clone(),
        config: TransactionConfig {
            skip_user_accounts_rpc_calls: false,
            wrap_and_unwrap_sol: wrap_unwrap,
            dynamic_compute_unit_limit: true,
            dynamic_slippage: Some(DynamicSlippageSettings { min_bps: Some(50), max_bps: Some(1200) }),
            ..TransactionConfig::default()
        },
    }).await?;

    let luts = get_address_lookup_table_accounts(rpc, response.address_lookup_table_addresses).await?;

    // Build full instruction list: compute budget + setup + swap + cleanup
    let mut ixs = vec![
        ComputeBudgetInstruction::set_compute_unit_limit(1_400_000),
        ComputeBudgetInstruction::set_compute_unit_price(1_000_000),
    ];
    // Include Jupiter-provided setup/cleanup instructions
    ixs.extend(response.setup_instructions.clone());
    // Always push the core swap instruction
    ixs.push(response.swap_instruction.clone());
    // Cleanup
    if let Some(ix) = response.cleanup_instruction.clone() {
        ixs.push(ix);
    }

    // Send as v0 with LUTs
    let blockhash = rpc.get_latest_blockhash().await?;
    let message = Message::try_compile(&payer.pubkey(), &ixs, &luts, blockhash)?;
    let tx = VersionedTransaction::try_new(VersionedMessage::V0(message), &[payer])?;

    match rpc.send_and_confirm_transaction(&tx).await {
        Ok(signature) => {
            println!("swap ore_in={} -> sol_out_est={} tx={}",
                fmt_ore(in_amount_grams),
                fmt_sol(out_amount_est),
                signature
            );
            Ok(())
        }
        Err(e) => {
            println!("Swap submit failed: {e:#?}");
            Err(e.into())
        }
    }
}

// Simulate a Jupiter swap between any two tokens (no submission).
async fn swap_sim(
	rpc: &RpcClient,
	payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
	let api_base_url = std::env::var("API_BASE_URL").unwrap_or("https://lite-api.jup.ag/swap/v1".into());
	let jupiter = JupiterSwapApiClient::new(api_base_url);

	let in_mint_str = std::env::var("IN_MINT").expect("Missing IN_MINT env var");
	let out_mint_str = std::env::var("OUT_MINT").expect("Missing OUT_MINT env var");
	let amount = std::env::var("AMOUNT").expect("Missing AMOUNT env var");
	let amount = u64::from_str(&amount).expect("Invalid AMOUNT");

	// Support shortcuts: SOL / ORE
	let parse_mint = |s: &str| -> Pubkey {
		match s {
			"SOL" | "So11111111111111111111111111111111111111112" => ore_api::consts::SOL_MINT,
			"ORE" | "oreoU2P8bN6jkk3jbaiVxYnG1dCXcYxwhwyK9jSybcp" => ore_api::consts::MINT_ADDRESS,
			_ => Pubkey::from_str(s).expect("Invalid mint pubkey"),
		}
	};
	let input_mint = parse_mint(&in_mint_str);
	let output_mint = parse_mint(&out_mint_str);

	let quote_request = QuoteRequest {
		amount,
		input_mint,
		output_mint,
		max_accounts: Some(55),
		..QuoteRequest::default()
	};

	let quote = jupiter.quote(&quote_request).await?;
	println!(
		"Quote: in_amount={} out_amount={} in_mint={} out_mint={}",
		amount,
		quote.out_amount,
		input_mint,
		output_mint
	);

	let wrap_unwrap = input_mint == ore_api::consts::SOL_MINT || output_mint == ore_api::consts::SOL_MINT;
	let response = jupiter
		.swap_instructions(&SwapRequest {
			user_public_key: payer.pubkey(),
			quote_response: quote.clone(),
			config: TransactionConfig {
				skip_user_accounts_rpc_calls: false,
				wrap_and_unwrap_sol: wrap_unwrap,
				dynamic_compute_unit_limit: true,
				dynamic_slippage: Some(DynamicSlippageSettings {
					min_bps: Some(50),
					max_bps: Some(1200),
				}),
				..TransactionConfig::default()
			},
		})
		.await?;

	let luts = get_address_lookup_table_accounts(rpc, response.address_lookup_table_addresses).await?;

	// Simulate with compute budget; include setup/swap/cleanup when provided
	let mut ixs = vec![
		ComputeBudgetInstruction::set_compute_unit_limit(1_400_000),
		ComputeBudgetInstruction::set_compute_unit_price(1_000_000),
	];
	// If available in the API, include setup and cleanup instructions
    ixs.extend(response.setup_instructions.clone());
	ixs.push(response.swap_instruction.clone());
    if let Some(ix) = response.cleanup_instruction.clone() {
        ixs.push(ix);
    }

	simulate_transaction_with_address_lookup_tables(rpc, payer, &ixs, luts).await;
	Ok(())
}

// Fetch and print ORE->SOL price using a 1 ORE quote
async fn price(
    _rpc: &RpcClient,
    _payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let api_base_url = std::env::var("API_BASE_URL").unwrap_or("https://lite-api.jup.ag/swap/v1".into());
    let jupiter = JupiterSwapApiClient::new(api_base_url);
    const INPUT_MINT: Pubkey = ore_api::consts::MINT_ADDRESS;
    const OUTPUT_MINT: Pubkey = ore_api::consts::SOL_MINT;
    let amount = ore_api::consts::ONE_ORE; // exactly 1 ORE in base units (grams)
    // Raw quote for diagnostics
    let req = QuoteRequest { amount, input_mint: INPUT_MINT, output_mint: OUTPUT_MINT, max_accounts: Some(55), ..QuoteRequest::default() };
    match jupiter.quote(&req).await {
        Ok(q) => {
            let lamports_out = q.out_amount as u64;
            let price_sol_per_ore = (lamports_out as f64) / (LAMPORTS_PER_SOL as f64);
            println!("QUOTE amount_in_ore={} out_lamports={} price_sol_per_ore={:.9}", 1, lamports_out, price_sol_per_ore);
        }
        Err(e) => {
            println!("QUOTE failed: {e:#?}");
        }
    }
    // Cached getter path
    match get_ore_sol_price(&jupiter).await {
        Ok(p) => println!("PRICE (cached-getter) sol_per_ore={:.9}", p),
        Err(e) => println!("PRICE (cached-getter) failed: {e:#?}"),
    }
    Ok(())
}

//

//

//

//

#[allow(dead_code)]
pub async fn get_address_lookup_table_accounts(
    rpc_client: &RpcClient,
    addresses: Vec<Pubkey>,
) -> Result<Vec<AddressLookupTableAccount>, anyhow::Error> {
    let mut accounts = Vec::new();
    for key in addresses {
        if let Ok(account) = rpc_client.get_account(&key).await {
            if let Ok(address_lookup_table_account) = AddressLookupTable::deserialize(&account.data)
            {
                accounts.push(AddressLookupTableAccount {
                    key,
                    addresses: address_lookup_table_account.addresses.to_vec(),
                });
            }
        }
    }
    Ok(accounts)
}

pub const ORE_VAR_ADDRESS: Pubkey = pubkey!("BWCaDY96Xe4WkFq1M7UiCCRcChsJ3p51L5KrGzhxgm2E");

async fn reset(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let board = get_board(rpc).await?;
    let var = get_var(rpc, ORE_VAR_ADDRESS).await?;

    println!("Var: {:?}", var);

    let client = reqwest::Client::new();
    let url = format!("https://entropy-api.onrender.com/var/{ORE_VAR_ADDRESS}/seed");
    let response = client
        .get(url)
        .send()
        .await?
        .json::<entropy_types::response::GetSeedResponse>()
        .await?;
    println!("Entropy seed: {:?}", response);

    let config = get_config(rpc).await?;
    let sample_ix = entropy_api::sdk::sample(payer.pubkey(), ORE_VAR_ADDRESS);
    let reveal_ix = entropy_api::sdk::reveal(payer.pubkey(), ORE_VAR_ADDRESS, response.seed);
    let reset_ix = ore_api::sdk::reset(
        payer.pubkey(),
        config.fee_collector,
        board.round_id,
        Pubkey::default(),
    );
    let sig = submit_transaction(rpc, payer, &[sample_ix, reveal_ix, reset_ix]).await?;
    println!("Reset: {}", sig);

    // let slot_hashes = get_slot_hashes(rpc).await?;
    // if let Some(slot_hash) = slot_hashes.get(&board.end_slot) {
    //     let id = get_winning_square(&slot_hash.to_bytes());
    //     // let square = get_square(rpc).await?;
    //     println!("Winning square: {}", id);
    //     // println!("Miners: {:?}", square.miners);
    //     // miners = square.miners[id as usize].to_vec();
    // };

    // let reset_ix = ore_api::sdk::reset(
    //     payer.pubkey(),
    //     config.fee_collector,
    //     board.round_id,
    //     Pubkey::default(),
    // );
    // // simulate_transaction(rpc, payer, &[reset_ix]).await;
    // submit_transaction(rpc, payer, &[reset_ix]).await?;
    Ok(())
}

async fn deploy(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let amount = std::env::var("AMOUNT").expect("Missing AMOUNT env var");
    let amount = u64::from_str(&amount).expect("Invalid AMOUNT");
    let square_id = std::env::var("SQUARE").expect("Missing SQUARE env var");
    let square_id = u64::from_str(&square_id).expect("Invalid SQUARE");
    let board = get_board(rpc).await?;
    let mut squares = [false; 25];
    squares[square_id as usize] = true;
    let ix = ore_api::sdk::deploy(
        payer.pubkey(),
        payer.pubkey(),
        amount,
        board.round_id,
        squares,
    );
    submit_transaction(rpc, payer, &[ix]).await?;
    Ok(())
}

async fn deploy_all(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let amount = std::env::var("AMOUNT").expect("Missing AMOUNT env var");
    let amount = u64::from_str(&amount).expect("Invalid AMOUNT");
    let board = get_board(rpc).await?;
    let squares = [true; 25];
    let ix = ore_api::sdk::deploy(
        payer.pubkey(),
        payer.pubkey(),
        board.round_id,
        amount,
        squares,
    );
    submit_transaction(rpc, payer, &[ix]).await?;
    Ok(())
}

async fn set_admin(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let ix = ore_api::sdk::set_admin(payer.pubkey(), payer.pubkey());
    submit_transaction(rpc, payer, &[ix]).await?;
    Ok(())
}

async fn set_swap_program(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let swap_program = std::env::var("SWAP_PROGRAM").expect("Missing SWAP_PROGRAM env var");
    let swap_program = Pubkey::from_str(&swap_program).expect("Invalid SWAP_PROGRAM");
    let ix = ore_api::sdk::set_swap_program(payer.pubkey(), swap_program);
    submit_transaction(rpc, payer, &[ix]).await?;
    Ok(())
}

async fn set_fee_collector(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let fee_collector = std::env::var("FEE_COLLECTOR").expect("Missing FEE_COLLECTOR env var");
    let fee_collector = Pubkey::from_str(&fee_collector).expect("Invalid FEE_COLLECTOR");
    let ix = ore_api::sdk::set_fee_collector(payer.pubkey(), fee_collector);
    submit_transaction(rpc, payer, &[ix]).await?;
    Ok(())
}

async fn checkpoint(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let authority = std::env::var("AUTHORITY").unwrap_or(payer.pubkey().to_string());
    let authority = Pubkey::from_str(&authority).expect("Invalid AUTHORITY");
    let miner = get_miner(rpc, authority).await?;
    let ix = ore_api::sdk::checkpoint(payer.pubkey(), authority, miner.round_id);
    submit_transaction(rpc, payer, &[ix]).await?;
    Ok(())
}

async fn checkpoint_all(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let clock = get_clock(rpc).await?;
    let miners = get_miners(rpc).await?;
    let mut expiry_slots = HashMap::new();
    let mut ixs = vec![];
    for (i, (_address, miner)) in miners.iter().enumerate() {
        if miner.checkpoint_id < miner.round_id {
            // Log the expiry slot for the round.
            if !expiry_slots.contains_key(&miner.round_id) {
                if let Ok(round) = get_round(rpc, miner.round_id).await {
                    expiry_slots.insert(miner.round_id, round.expires_at);
                }
            }

            // Get the expiry slot for the round.
            let Some(expires_at) = expiry_slots.get(&miner.round_id) else {
                continue;
            };

            // If we are in fee collection period, checkpoint the miner.
            if clock.slot >= expires_at - TWELVE_HOURS_SLOTS {
                println!(
                    "[{}/{}] Checkpoint miner: {} ({} slots left)",
                    i + 1,
                    miners.len(),
                    miner.authority,
                    expires_at.saturating_sub(clock.slot)
                );
                ixs.push(ore_api::sdk::checkpoint(
                    payer.pubkey(),
                    miner.authority,
                    miner.round_id,
                ));
            }
        }
    }

    // Batch and submit the instructions.
    while !ixs.is_empty() {
        let batch = ixs
            .drain(..std::cmp::min(10, ixs.len()))
            .collect::<Vec<Instruction>>();
        submit_transaction(rpc, payer, &batch).await?;
    }

    Ok(())
}

async fn close_all(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let rounds = get_rounds(rpc).await?;
    let mut ixs = vec![];
    let clock = get_clock(rpc).await?;
    for (_i, (_address, round)) in rounds.iter().enumerate() {
        if clock.slot >= round.expires_at {
            ixs.push(ore_api::sdk::close(
                payer.pubkey(),
                round.id,
                round.rent_payer,
            ));
        }
    }

    // Batch and submit the instructions.
    while !ixs.is_empty() {
        let batch = ixs
            .drain(..std::cmp::min(12, ixs.len()))
            .collect::<Vec<Instruction>>();
        // simulate_transaction(rpc, payer, &batch).await;
        submit_transaction(rpc, payer, &batch).await?;
    }

    Ok(())
}

// async fn log_meteora_pool(rpc: &RpcClient) -> Result<(), anyhow::Error> {
//     let address = pubkey!("GgaDTFbqdgjoZz3FP7zrtofGwnRS4E6MCzmmD5Ni1Mxj");
//     let pool = get_meteora_pool(rpc, address).await?;
//     let vault_a = get_meteora_vault(rpc, pool.a_vault).await?;
//     let vault_b = get_meteora_vault(rpc, pool.b_vault).await?;

//     println!("Pool");
//     println!("  address: {}", address);
//     println!("  lp_mint: {}", pool.lp_mint);
//     println!("  token_a_mint: {}", pool.token_a_mint);
//     println!("  token_b_mint: {}", pool.token_b_mint);
//     println!("  a_vault: {}", pool.a_vault);
//     println!("  b_vault: {}", pool.b_vault);
//     println!("  a_token_vault: {}", vault_a.token_vault);
//     println!("  b_token_vault: {}", vault_b.token_vault);
//     println!("  a_vault_lp_mint: {}", vault_a.lp_mint);
//     println!("  b_vault_lp_mint: {}", vault_b.lp_mint);
//     println!("  a_vault_lp: {}", pool.a_vault_lp);
//     println!("  b_vault_lp: {}", pool.b_vault_lp);
//     println!("  protocol_token_fee: {}", pool.protocol_token_b_fee);

//     // pool: *pool.key,
//     // user_source_token: *user_source_token.key,
//     // user_destination_token: *user_destination_token.key,
//     // a_vault: *a_vault.key,
//     // b_vault: *b_vault.key,
//     // a_token_vault: *a_token_vault.key,
//     // b_token_vault: *b_token_vault.key,
//     // a_vault_lp_mint: *a_vault_lp_mint.key,
//     // b_vault_lp_mint: *b_vault_lp_mint.key,
//     // a_vault_lp: *a_vault_lp.key,
//     // b_vault_lp: *b_vault_lp.key,
//     // protocol_token_fee: *protocol_token_fee.key,
//     // user: *user.key,
//     // vault_program: *vault_program.key,
//     // token_program: *token_program.key,

//     Ok(())
// }

async fn log_automations(rpc: &RpcClient) -> Result<(), anyhow::Error> {
    let automations = get_automations(rpc).await?;
    for (i, (address, automation)) in automations.iter().enumerate() {
        println!("[{}/{}] {}", i + 1, automations.len(), address);
        println!("  authority: {}", automation.authority);
        println!("  balance: {}", automation.balance);
        println!("  executor: {}", automation.executor);
        println!("  fee: {}", automation.fee);
        println!("  mask: {}", automation.mask);
        println!("  strategy: {}", automation.strategy);
        println!();
    }
    Ok(())
}

async fn log_treasury(rpc: &RpcClient) -> Result<(), anyhow::Error> {
    let treasury_address = ore_api::state::treasury_pda().0;
    let treasury = get_treasury(rpc).await?;
    println!("Treasury");
    println!("  address: {}", treasury_address);
    println!("  balance: {} SOL", lamports_to_sol(treasury.balance));
    println!(
        "  motherlode: {} ORE",
        amount_to_ui_amount(treasury.motherlode, TOKEN_DECIMALS)
    );
    println!(
        "  miner_rewards_factor: {}",
        treasury.miner_rewards_factor.to_i80f48().to_string()
    );
    println!(
        "  stake_rewards_factor: {}",
        treasury.stake_rewards_factor.to_i80f48().to_string()
    );
    println!(
        "  total_staked: {} ORE",
        amount_to_ui_amount(treasury.total_staked, TOKEN_DECIMALS)
    );
    println!(
        "  total_unclaimed: {} ORE",
        amount_to_ui_amount(treasury.total_unclaimed, TOKEN_DECIMALS)
    );
    println!(
        "  total_refined: {} ORE",
        amount_to_ui_amount(treasury.total_refined, TOKEN_DECIMALS)
    );
    Ok(())
}

async fn log_round(rpc: &RpcClient) -> Result<(), anyhow::Error> {
    let id = std::env::var("ID").expect("Missing ID env var");
    let id = u64::from_str(&id).expect("Invalid ID");
    let round_address = round_pda(id).0;
    let round = get_round(rpc, id).await?;
    let rng = round.rng();
    println!("Round");
    println!("  Address: {}", round_address);
    println!("  Count: {:?}", round.count);
    println!("  Deployed: {:?}", round.deployed);
    println!("  Expires at: {}", round.expires_at);
    println!("  Id: {:?}", round.id);
    println!("  Motherlode: {}", round.motherlode);
    println!("  Rent payer: {}", round.rent_payer);
    println!("  Slot hash: {:?}", round.slot_hash);
    println!("  Top miner: {:?}", round.top_miner);
    println!("  Top miner reward: {}", round.top_miner_reward);
    println!("  Total deployed: {}", round.total_deployed);
    println!("  Total vaulted: {}", round.total_vaulted);
    println!("  Total winnings: {}", round.total_winnings);
    if let Some(rng) = rng {
        println!("  Winning square: {}", round.winning_square(rng));
    }
    // if round.slot_hash != [0; 32] {
    //     println!("  Winning square: {}", get_winning_square(&round.slot_hash));
    // }
    Ok(())
}

async fn log_miner(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
) -> Result<(), anyhow::Error> {
    let authority = std::env::var("AUTHORITY").unwrap_or(payer.pubkey().to_string());
    let authority = Pubkey::from_str(&authority).expect("Invalid AUTHORITY");
    let miner_address = ore_api::state::miner_pda(authority).0;
    let miner = get_miner(&rpc, authority).await?;
    println!("Miner");
    println!("  address: {}", miner_address);
    println!("  authority: {}", authority);
    println!("  deployed: {:?}", miner.deployed);
    println!("  cumulative: {:?}", miner.cumulative);
    println!("  rewards_sol: {} SOL", lamports_to_sol(miner.rewards_sol));
    println!(
        "  rewards_ore: {} ORE",
        amount_to_ui_amount(miner.rewards_ore, TOKEN_DECIMALS)
    );
    println!(
        "  refined_ore: {} ORE",
        amount_to_ui_amount(miner.refined_ore, TOKEN_DECIMALS)
    );
    println!("  round_id: {}", miner.round_id);
    println!("  checkpoint_id: {}", miner.checkpoint_id);
    println!(
        "  lifetime_rewards_sol: {} SOL",
        lamports_to_sol(miner.lifetime_rewards_sol)
    );
    println!(
        "  lifetime_rewards_ore: {} ORE",
        amount_to_ui_amount(miner.lifetime_rewards_ore, TOKEN_DECIMALS)
    );
    Ok(())
}

async fn log_clock(rpc: &RpcClient) -> Result<(), anyhow::Error> {
    let clock = get_clock(&rpc).await?;
    println!("Clock");
    println!("  slot: {}", clock.slot);
    println!("  epoch_start_timestamp: {}", clock.epoch_start_timestamp);
    println!("  epoch: {}", clock.epoch);
    println!("  leader_schedule_epoch: {}", clock.leader_schedule_epoch);
    println!("  unix_timestamp: {}", clock.unix_timestamp);
    Ok(())
}

async fn log_config(rpc: &RpcClient) -> Result<(), anyhow::Error> {
    let config = get_config(&rpc).await?;
    println!("Config");
    println!("  admin: {}", config.admin);
    println!("  bury_authority: {}", config.bury_authority);
    println!("  fee_collector: {}", config.fee_collector);
    println!("  swap_program: {}", config.swap_program);
    println!("  buffer: {}", config.buffer);

    Ok(())
}

async fn log_board(rpc: &RpcClient) -> Result<(), anyhow::Error> {
    let board = get_board(&rpc).await?;
    let clock = get_clock(&rpc).await?;
    print_board(board, &clock);
    Ok(())
}

fn print_board(board: Board, clock: &Clock) {
    let current_slot = clock.slot;
    println!("Board");
    println!("  Id: {:?}", board.round_id);
    println!("  Start slot: {}", board.start_slot);
    println!("  End slot: {}", board.end_slot);
    println!("  Time remaining (slots): {}", board.end_slot.saturating_sub(current_slot));
}

async fn get_automations(rpc: &RpcClient) -> Result<Vec<(Pubkey, Automation)>, anyhow::Error> {
    const REGOLITH_EXECUTOR: Pubkey = pubkey!("HNWhK5f8RMWBqcA7mXJPaxdTPGrha3rrqUrri7HSKb3T");
    let filter = RpcFilterType::Memcmp(Memcmp::new_base58_encoded(
        56,
        &REGOLITH_EXECUTOR.to_bytes(),
    ));
    let automations = get_program_accounts::<Automation>(rpc, ore_api::ID, vec![filter]).await?;
    Ok(automations)
}

// async fn get_meteora_pool(rpc: &RpcClient, address: Pubkey) -> Result<Pool, anyhow::Error> {
//     let data = rpc.get_account_data(&address).await?;
//     let pool = Pool::from_bytes(&data)?;
//     Ok(pool)
// }

// async fn get_meteora_vault(rpc: &RpcClient, address: Pubkey) -> Result<Vault, anyhow::Error> {
//     let data = rpc.get_account_data(&address).await?;
//     let vault = Vault::from_bytes(&data)?;
//     Ok(vault)
// }

async fn get_board(rpc: &RpcClient) -> Result<Board, anyhow::Error> {
    let board_pda = ore_api::state::board_pda();
    let account = rpc.get_account(&board_pda.0).await?;
    let board = Board::try_from_bytes(&account.data)?;
    Ok(*board)
}

async fn get_var(rpc: &RpcClient, address: Pubkey) -> Result<Var, anyhow::Error> {
    let account = rpc.get_account(&address).await?;
    let var = Var::try_from_bytes(&account.data)?;
    Ok(*var)
}

async fn get_round(rpc: &RpcClient, id: u64) -> Result<Round, anyhow::Error> {
    let round_pda = ore_api::state::round_pda(id);
    let account = rpc.get_account(&round_pda.0).await?;
    let round = Round::try_from_bytes(&account.data)?;
    Ok(*round)
}

async fn get_treasury(rpc: &RpcClient) -> Result<Treasury, anyhow::Error> {
    let treasury_pda = ore_api::state::treasury_pda();
    let account = rpc.get_account(&treasury_pda.0).await?;
    let treasury = Treasury::try_from_bytes(&account.data)?;
    Ok(*treasury)
}

async fn get_config(rpc: &RpcClient) -> Result<Config, anyhow::Error> {
    let config_pda = ore_api::state::config_pda();
    let account = rpc.get_account(&config_pda.0).await?;
    let config = Config::try_from_bytes(&account.data)?;
    Ok(*config)
}

async fn get_miner(rpc: &RpcClient, authority: Pubkey) -> Result<Miner, anyhow::Error> {
    let miner_pda = ore_api::state::miner_pda(authority);
    let account = rpc.get_account(&miner_pda.0).await?;
    let miner = Miner::try_from_bytes(&account.data)?;
    Ok(*miner)
}

async fn get_clock(rpc: &RpcClient) -> Result<Clock, anyhow::Error> {
    let data = rpc.get_account_data(&solana_sdk::sysvar::clock::ID).await?;
    let clock = bincode::deserialize::<Clock>(&data)?;
    Ok(clock)
}

async fn get_stake(rpc: &RpcClient, authority: Pubkey) -> Result<Stake, anyhow::Error> {
    let stake_pda = ore_api::state::stake_pda(authority);
    let account = rpc.get_account(&stake_pda.0).await?;
    let stake = Stake::try_from_bytes(&account.data)?;
    Ok(*stake)
}

async fn get_rounds(rpc: &RpcClient) -> Result<Vec<(Pubkey, Round)>, anyhow::Error> {
    let rounds = get_program_accounts::<Round>(rpc, ore_api::ID, vec![]).await?;
    Ok(rounds)
}

#[allow(dead_code)]
async fn get_miners(rpc: &RpcClient) -> Result<Vec<(Pubkey, Miner)>, anyhow::Error> {
    let miners = get_program_accounts::<Miner>(rpc, ore_api::ID, vec![]).await?;
    Ok(miners)
}

async fn get_miners_participating(
    rpc: &RpcClient,
    round_id: u64,
) -> Result<Vec<(Pubkey, Miner)>, anyhow::Error> {
    let filter = RpcFilterType::Memcmp(Memcmp::new_base58_encoded(512, &round_id.to_le_bytes()));
    let miners = get_program_accounts::<Miner>(rpc, ore_api::ID, vec![filter]).await?;
    Ok(miners)
}

// fn get_winning_square(slot_hash: &[u8]) -> u64 {
//     // Use slot hash to generate a random u64
//     let r1 = u64::from_le_bytes(slot_hash[0..8].try_into().unwrap());
//     let r2 = u64::from_le_bytes(slot_hash[8..16].try_into().unwrap());
//     let r3 = u64::from_le_bytes(slot_hash[16..24].try_into().unwrap());
//     let r4 = u64::from_le_bytes(slot_hash[24..32].try_into().unwrap());
//     let r = r1 ^ r2 ^ r3 ^ r4;
//     // Returns a value in the range [0, 24] inclusive
//     r % 25
// }

#[allow(dead_code)]
async fn simulate_transaction(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
    instructions: &[solana_sdk::instruction::Instruction],
) {
    let blockhash = rpc.get_latest_blockhash().await.unwrap();
    let x = rpc
        .simulate_transaction(&Transaction::new_signed_with_payer(
            instructions,
            Some(&payer.pubkey()),
            &[payer],
            blockhash,
        ))
        .await;
    println!("Simulation result: {:?}", x);
}

async fn simulate_with_budget(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
    inner_ixs: &[solana_sdk::instruction::Instruction],
) {
    let mut ixs = vec![
        ComputeBudgetInstruction::set_compute_unit_limit(1_400_000),
        ComputeBudgetInstruction::set_compute_unit_price(1_000_000),
    ];
    ixs.extend_from_slice(inner_ixs);
    let blockhash = rpc.get_latest_blockhash().await.unwrap();
    let sim = rpc
        .simulate_transaction(&Transaction::new_signed_with_payer(
            &ixs,
            Some(&payer.pubkey()),
            &[payer],
            blockhash,
        ))
        .await;
    println!("simulate: {:?}", sim);
}

async fn simulate_for_ok(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
    inner_ixs: &[solana_sdk::instruction::Instruction],
) -> Result<bool, anyhow::Error> {
    let mut ixs = vec![
        ComputeBudgetInstruction::set_compute_unit_limit(1_400_000),
        ComputeBudgetInstruction::set_compute_unit_price(1_000_000),
    ];
    ixs.extend_from_slice(inner_ixs);
    let blockhash = rpc.get_latest_blockhash().await?;
    let sim = rpc
        .simulate_transaction(&Transaction::new_signed_with_payer(
            &ixs,
            Some(&payer.pubkey()),
            &[payer],
            blockhash,
        ))
        .await?;
    Ok(sim.value.err.is_none())
}

// Simulate using cached blockhash to avoid repeated latest_blockhash RPC calls
async fn simulate_for_ok_cached(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
    inner_ixs: &[solana_sdk::instruction::Instruction],
    bh_cache: &BlockhashCache,
    current_slot_opt: Option<u64>,
) -> Result<bool, anyhow::Error> {
    let mut ixs = vec![
        ComputeBudgetInstruction::set_compute_unit_limit(1_400_000),
        ComputeBudgetInstruction::set_compute_unit_price(1_000_000),
    ];
    ixs.extend_from_slice(inner_ixs);
    let blockhash = bh_cache.get_fresh(rpc, current_slot_opt).await?;
    let sim = rpc
        .simulate_transaction(&Transaction::new_signed_with_payer(
            &ixs,
            Some(&payer.pubkey()),
            &[payer],
            blockhash,
        ))
        .await?;
    Ok(sim.value.err.is_none())
}

// Submit a transaction and confirm via WebSocket signature subscription
async fn submit_transaction_ws_confirm_cached(
    rpc: &RpcClient,
    ws_url: &str,
    payer: &solana_sdk::signer::keypair::Keypair,
    inner_ixs: &[solana_sdk::instruction::Instruction],
    bh_cache: &BlockhashCache,
    current_slot_opt: Option<u64>,
) -> Result<solana_sdk::signature::Signature, anyhow::Error> {
    let mut all_instructions = vec![
        ComputeBudgetInstruction::set_compute_unit_limit(1_400_000),
        ComputeBudgetInstruction::set_compute_unit_price(1_000_000),
    ];
    all_instructions.extend_from_slice(inner_ixs);

    let blockhash = bh_cache.get_fresh(rpc, current_slot_opt).await?;
    let transaction = Transaction::new_signed_with_payer(
        &all_instructions,
        Some(&payer.pubkey()),
        &[payer],
        blockhash,
    );
    let sig = rpc.send_transaction(&transaction).await?;

    // Confirm via PubsubClient
    let client = PubsubClient::new(ws_url).await?;
    let (mut stream, _unsub) = client.signature_subscribe(&sig, None).await?;
    // Break on first WS confirmation event (then verify status via HTTP once)
    while let Some(_update) = stream.next().await {
        break;
    }
    // Verify status once (single HTTP call)
    if let Ok(statuses) = rpc.get_signature_statuses(&[sig]).await {
        if let Some(Some(st)) = statuses.value.get(0).map(|o| o.as_ref()) {
            if st.err.is_some() {
                return Err(anyhow::anyhow!("transaction failed: {:?}", st.err));
            }
        }
    }
    Ok(sig)
}

#[allow(dead_code)]
async fn simulate_transaction_with_address_lookup_tables(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
    instructions: &[solana_sdk::instruction::Instruction],
    address_lookup_table_accounts: Vec<AddressLookupTableAccount>,
) {
    let blockhash = rpc.get_latest_blockhash().await.unwrap();
    let tx = VersionedTransaction {
        signatures: vec![Signature::default()],
        message: VersionedMessage::V0(
            Message::try_compile(
                &payer.pubkey(),
                instructions,
                &address_lookup_table_accounts,
                blockhash,
            )
            .unwrap(),
        ),
    };
    let s = tx.sanitize();
    println!("Sanitize result: {:?}", s);
    s.unwrap();
    let x = rpc.simulate_transaction(&tx).await;
    println!("Simulation result: {:?}", x);
}

#[allow(unused)]
async fn submit_transaction_batches(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
    mut ixs: Vec<solana_sdk::instruction::Instruction>,
    batch_size: usize,
) -> Result<(), anyhow::Error> {
    // Batch and submit the instructions.
    while !ixs.is_empty() {
        let batch = ixs
            .drain(..std::cmp::min(batch_size, ixs.len()))
            .collect::<Vec<Instruction>>();
        submit_transaction_no_confirm(rpc, payer, &batch).await?;
    }
    Ok(())
}

#[allow(unused)]
async fn simulate_transaction_batches(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
    mut ixs: Vec<solana_sdk::instruction::Instruction>,
    batch_size: usize,
) -> Result<(), anyhow::Error> {
    // Batch and submit the instructions.
    while !ixs.is_empty() {
        let batch = ixs
            .drain(..std::cmp::min(batch_size, ixs.len()))
            .collect::<Vec<Instruction>>();
        simulate_transaction(rpc, payer, &batch).await;
    }
    Ok(())
}

async fn submit_transaction(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
    instructions: &[solana_sdk::instruction::Instruction],
) -> Result<solana_sdk::signature::Signature, anyhow::Error> {
    let blockhash = rpc.get_latest_blockhash().await?;
    let mut all_instructions = vec![
        ComputeBudgetInstruction::set_compute_unit_limit(1_400_000),
        ComputeBudgetInstruction::set_compute_unit_price(1_000_000),
    ];
    all_instructions.extend_from_slice(instructions);
    let transaction = Transaction::new_signed_with_payer(
        &all_instructions,
        Some(&payer.pubkey()),
        &[payer],
        blockhash,
    );

    match rpc.send_and_confirm_transaction(&transaction).await {
        Ok(signature) => {
            println!("Transaction submitted: {:?}", signature);
            Ok(signature)
        }
        Err(e) => {
            println!("Error submitting transaction: {:?}", e);
            Err(e.into())
        }
    }
}

async fn submit_transaction_no_confirm(
    rpc: &RpcClient,
    payer: &solana_sdk::signer::keypair::Keypair,
    instructions: &[solana_sdk::instruction::Instruction],
) -> Result<solana_sdk::signature::Signature, anyhow::Error> {
    let blockhash = rpc.get_latest_blockhash().await?;
    let mut all_instructions = vec![
        ComputeBudgetInstruction::set_compute_unit_limit(1_400_000),
        ComputeBudgetInstruction::set_compute_unit_price(1_000_000),
    ];
    all_instructions.extend_from_slice(instructions);
    let transaction = Transaction::new_signed_with_payer(
        &all_instructions,
        Some(&payer.pubkey()),
        &[payer],
        blockhash,
    );

    match rpc.send_transaction(&transaction).await {
        Ok(signature) => {
            println!("Transaction submitted: {:?}", signature);
            Ok(signature)
        }
        Err(e) => {
            println!("Error submitting transaction: {:?}", e);
            Err(e.into())
        }
    }
}

pub async fn get_program_accounts<T>(
    client: &RpcClient,
    program_id: Pubkey,
    filters: Vec<RpcFilterType>,
) -> Result<Vec<(Pubkey, T)>, anyhow::Error>
where
    T: AccountDeserialize + Discriminator + Clone,
{
    let mut all_filters = vec![RpcFilterType::Memcmp(Memcmp::new_base58_encoded(
        0,
        &T::discriminator().to_le_bytes(),
    ))];
    all_filters.extend(filters);
    let result = client
        .get_program_accounts_with_config(
            &program_id,
            RpcProgramAccountsConfig {
                filters: Some(all_filters),
                account_config: RpcAccountInfoConfig {
                    encoding: Some(UiAccountEncoding::Base64),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .await;

    match result {
        Ok(accounts) => {
            let accounts = accounts
                .into_iter()
                .filter_map(|(pubkey, account)| {
                    if let Ok(account) = T::try_from_bytes(&account.data) {
                        Some((pubkey, account.clone()))
                    } else {
                        None
                    }
                })
                .collect();
            Ok(accounts)
        }
        Err(err) => match err.kind {
            ClientErrorKind::Reqwest(err) => {
                if let Some(status_code) = err.status() {
                    if status_code == StatusCode::GONE {
                        panic!(
                                "\n{} Your RPC provider does not support the getProgramAccounts endpoint, needed to execute this command. Please use a different RPC provider.\n",
                                "ERROR"
                            );
                    }
                }
                return Err(anyhow::anyhow!("Failed to get program accounts: {}", err));
            }
            _ => return Err(anyhow::anyhow!("Failed to get program accounts: {}", err)),
        },
    }
}
