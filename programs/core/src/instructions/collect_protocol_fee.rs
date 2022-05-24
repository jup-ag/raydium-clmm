use crate::error::ErrorCode;
use crate::states::*;
use crate::util::*;
use anchor_lang::prelude::*;
use anchor_spl::token::{Token, TokenAccount};

#[derive(Accounts)]
pub struct CollectProtocolFee<'info> {
    /// Valid protocol owner
    #[account(address = factory_state.owner)]
    pub owner: Signer<'info>,

    /// Factory state stores the protocol owner address
    #[account(mut)]
    pub factory_state: Account<'info, FactoryState>,

    /// Pool state stores accumulated protocol fee amount
    #[account(mut)]
    pub pool_state: Box<Account<'info, PoolState>>,

    /// The address that holds pool tokens for token_0
    #[account(
        mut,
        token::mint = pool_state.token_0,
        constraint = vault_0.key() == pool_state.token_vault_0,
    )]
    pub vault_0: Account<'info, TokenAccount>,

    /// The address that holds pool tokens for token_1
    #[account(
        mut,
        token::mint = pool_state.token_1,
        constraint = vault_1.key() == pool_state.token_vault_1,
    )]
    pub vault_1: Account<'info, TokenAccount>,

    /// The address that receives the collected token_0 protocol fees
    #[account(mut)]
    pub recipient_wallet_0: Account<'info, TokenAccount>,

    /// The address that receives the collected token_1 protocol fees
    #[account(mut)]
    pub recipient_wallet_1: Account<'info, TokenAccount>,

    /// The SPL program to perform token transfers
    pub token_program: Program<'info, Token>,
}

pub fn collect_protocol_fee(
    ctx: Context<CollectProtocolFee>,
    amount_0_requested: u64,
    amount_1_requested: u64,
) -> Result<()> {
    let pool_state_info = ctx.accounts.pool_state.to_account_info();
    let pool_state = ctx.accounts.pool_state.as_mut();
    require!(pool_state.unlocked, ErrorCode::LOK);
    pool_state.unlocked = false;

    let amount_0 = amount_0_requested.min(pool_state.protocol_fees_token_0);
    let amount_1 = amount_1_requested.min(pool_state.protocol_fees_token_1);

    pool_state.protocol_fees_token_0 -= amount_0;
    pool_state.protocol_fees_token_1 -= amount_1;

    if amount_0 > 0 {
        transfer_from_pool_vault_to_user(
            pool_state,
            &ctx.accounts.vault_0,
            &ctx.accounts.recipient_wallet_0,
            &ctx.accounts.token_program,
            amount_0,
        )?;
    }
    if amount_1 > 0 {
        transfer_from_pool_vault_to_user(
            pool_state,
            &ctx.accounts.vault_1,
            &ctx.accounts.recipient_wallet_1,
            &ctx.accounts.token_program,
            amount_1,
        )?;
    }

    emit!(CollectProtocolEvent {
        pool_state: pool_state_info.key(),
        sender: ctx.accounts.owner.key(),
        recipient_wallet_0: ctx.accounts.recipient_wallet_0.key(),
        recipient_wallet_1: ctx.accounts.recipient_wallet_1.key(),
        amount_0,
        amount_1,
    });

    pool_state.unlocked = true;
    Ok(())
}