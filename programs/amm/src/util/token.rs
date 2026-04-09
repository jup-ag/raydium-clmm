use crate::states::*;
use anchor_lang::prelude::*;
use anchor_lang::solana_program::program::{invoke, invoke_signed};
use anchor_spl::{
    token::{self, Token},
    token_2022::{
        self,
        spl_token_2022::{
            self,
            extension::{
                transfer_fee::{TransferFeeConfig, MAX_FEE_BASIS_POINTS},
                BaseStateWithExtensions, ExtensionType, StateWithExtensions,
            },
        },
    },
    token_interface::{Mint, TokenAccount},
};
use std::collections::HashSet;

use super::get_recent_epoch;

const MINT_WHITELIST: [&'static str; 4] = [
    "HVbpJAQGNpkgBaYBZQBR1t7yFdvaYVp2vCQQfKKEN4tM",
    "Crn4x1Y2HUKko7ox2EZMT6N2t2ZyH7eKtwkBGVnhEq1g",
    "FrBfWJ4qE5sCzKm3k3JaAtqZcXUh4LvJygDeketsrsH4",
    "2b1kV6DkPAnxd5ixfnxCpjxmKwqjjaYmCZfHsFu24GXo",
];

pub fn invoke_memo_instruction<'info>(
    memo_msg: &[u8],
    memo_program: AccountInfo<'info>,
) -> solana_program::entrypoint::ProgramResult {
    let ix = spl_memo_interface::instruction::build_memo(
        &spl_memo_interface::v3::id(),
        memo_msg,
        &[],
    );
    let accounts = vec![memo_program];
    invoke(&ix, &accounts[..])
}

pub fn transfer_from_user_to_pool_vault<'info>(
    signer: &Signer<'info>,
    from: &InterfaceAccount<'info, TokenAccount>,
    to_vault: &InterfaceAccount<'info, TokenAccount>,
    mint: Option<Box<InterfaceAccount<'info, Mint>>>,
    token_program: &AccountInfo<'info>,
    token_program_2022: Option<AccountInfo<'info>>,
    amount: u64,
) -> Result<()> {
    if amount == 0 {
        return Ok(());
    }
    let token_program_info = token_program.to_account_info();
    let from_token_info = from.to_account_info();
    let to_vault_info = to_vault.to_account_info();
    let signer_info = signer.to_account_info();
    match (mint, token_program_2022) {
        (Some(mint), Some(token_program_2022)) => {
            if from_token_info.owner == token_program_2022.key {
                let mint_info = mint.to_account_info();
                let ix = spl_token_2022::instruction::transfer_checked(
                    token_program_2022.key,
                    from_token_info.key,
                    mint_info.key,
                    to_vault_info.key,
                    signer_info.key,
                    &[],
                    amount,
                    mint.decimals,
                )?;
                invoke(
                    &ix,
                    &[from_token_info, mint_info, to_vault_info, signer_info, token_program_2022],
                )
                .map_err(Into::into)
            } else {
                let mint_info = mint.to_account_info();
                let ix = token::spl_token::instruction::transfer_checked(
                    &token::ID,
                    from_token_info.key,
                    mint_info.key,
                    to_vault_info.key,
                    signer_info.key,
                    &[],
                    amount,
                    mint.decimals,
                )?;
                invoke(
                    &ix,
                    &[from_token_info, mint_info, to_vault_info, signer_info, token_program_info],
                )
                .map_err(Into::into)
            }
        }
        _ => {
            let ix = token::spl_token::instruction::transfer(
                &token::ID,
                from_token_info.key,
                to_vault_info.key,
                signer_info.key,
                &[],
                amount,
            )?;
            invoke(
                &ix,
                &[from_token_info, to_vault_info, signer_info, token_program_info],
            )
            .map_err(Into::into)
        }
    }
}

pub fn transfer_from_pool_vault_to_user<'info>(
    pool_state_loader: &AccountLoader<'info, PoolState>,
    from_vault: &InterfaceAccount<'info, TokenAccount>,
    to: &InterfaceAccount<'info, TokenAccount>,
    mint: Option<Box<InterfaceAccount<'info, Mint>>>,
    token_program: &AccountInfo<'info>,
    token_program_2022: Option<AccountInfo<'info>>,
    amount: u64,
) -> Result<()> {
    if amount == 0 {
        return Ok(());
    }
    let token_program_info = token_program.to_account_info();
    let from_vault_info = from_vault.to_account_info();
    let to_info = to.to_account_info();
    let authority_info = pool_state_loader.to_account_info();
    let pool_state = pool_state_loader.load()?;
    let seeds = pool_state.seeds();
    let signer_seeds: &[&[&[u8]]] = &[&seeds];
    match (mint, token_program_2022) {
        (Some(mint), Some(token_program_2022)) => {
            if from_vault_info.owner == token_program_2022.key {
                let mint_info = mint.to_account_info();
                let ix = spl_token_2022::instruction::transfer_checked(
                    token_program_2022.key,
                    from_vault_info.key,
                    mint_info.key,
                    to_info.key,
                    authority_info.key,
                    &[],
                    amount,
                    mint.decimals,
                )?;
                invoke_signed(
                    &ix,
                    &[from_vault_info, mint_info, to_info, authority_info, token_program_2022],
                    signer_seeds,
                )
                .map_err(Into::into)
            } else {
                let mint_info = mint.to_account_info();
                let ix = token::spl_token::instruction::transfer_checked(
                    &token::ID,
                    from_vault_info.key,
                    mint_info.key,
                    to_info.key,
                    authority_info.key,
                    &[],
                    amount,
                    mint.decimals,
                )?;
                invoke_signed(
                    &ix,
                    &[from_vault_info, mint_info, to_info, authority_info, token_program_info],
                    signer_seeds,
                )
                .map_err(Into::into)
            }
        }
        _ => {
            let ix = token::spl_token::instruction::transfer(
                &token::ID,
                from_vault_info.key,
                to_info.key,
                authority_info.key,
                &[],
                amount,
            )?;
            invoke_signed(
                &ix,
                &[from_vault_info, to_info, authority_info, token_program_info],
                signer_seeds,
            )
            .map_err(Into::into)
        }
    }
}

pub fn close_spl_account<'a, 'b, 'c, 'info>(
    owner: &AccountInfo<'info>,
    destination: &AccountInfo<'info>,
    close_account: &InterfaceAccount<'info, TokenAccount>,
    token_program: &Program<'info, Token>,
    // token_program_2022: &Program<'info, Token2022>,
    signers_seeds: &[&[&[u8]]],
) -> Result<()> {
    let close_account_info = close_account.to_account_info();
    let ix = token::spl_token::instruction::close_account(
        &token_program.key(),
        close_account_info.key,
        destination.key,
        owner.key,
        &[],
    )?;
    invoke_signed(
        &ix,
        &[close_account_info, destination.to_account_info(), owner.to_account_info()],
        signers_seeds,
    )
    .map_err(Into::into)
}

pub fn burn<'a, 'b, 'c, 'info>(
    owner: &Signer<'info>,
    mint: &InterfaceAccount<'info, Mint>,
    burn_account: &InterfaceAccount<'info, TokenAccount>,
    token_program: &Program<'info, Token>,
    // token_program_2022: &Program<'info, Token2022>,
    signers_seeds: &[&[&[u8]]],
    amount: u64,
) -> Result<()> {
    let mint_info = mint.to_account_info();
    let burn_account_info = burn_account.to_account_info();
    let owner_info = owner.to_account_info();
    let ix = token::spl_token::instruction::burn(
        &token_program.key(),
        burn_account_info.key,
        mint_info.key,
        owner_info.key,
        &[],
        amount,
    )?;
    invoke_signed(
        &ix,
        &[burn_account_info, mint_info, owner_info],
        signers_seeds,
    )
    .map_err(Into::into)
}

/// Calculate the fee for output amount
pub fn get_transfer_inverse_fee(
    mint_account: Box<InterfaceAccount<Mint>>,
    post_fee_amount: u64,
) -> Result<u64> {
    let mint_info = mint_account.to_account_info();
    if *mint_info.owner == Token::id() {
        return Ok(0);
    }
    let mint_data = mint_info.try_borrow_data()?;
    let mint = StateWithExtensions::<spl_token_2022::state::Mint>::unpack(&mint_data)?;

    let fee = if let Ok(transfer_fee_config) = mint.get_extension::<TransferFeeConfig>() {
        let epoch = get_recent_epoch()?;

        let transfer_fee = transfer_fee_config.get_epoch_fee(epoch);
        if u16::from(transfer_fee.transfer_fee_basis_points) == MAX_FEE_BASIS_POINTS {
            u64::from(transfer_fee.maximum_fee)
        } else {
            transfer_fee_config
                .calculate_inverse_epoch_fee(epoch, post_fee_amount)
                .unwrap()
        }
    } else {
        0
    };
    Ok(fee)
}

/// Calculate the fee for input amount
pub fn get_transfer_fee(
    mint_account: Box<InterfaceAccount<Mint>>,
    pre_fee_amount: u64,
) -> Result<u64> {
    let mint_info = mint_account.to_account_info();
    if *mint_info.owner == Token::id() {
        return Ok(0);
    }
    let mint_data = mint_info.try_borrow_data()?;
    let mint = StateWithExtensions::<spl_token_2022::state::Mint>::unpack(&mint_data)?;

    let fee = if let Ok(transfer_fee_config) = mint.get_extension::<TransferFeeConfig>() {
        transfer_fee_config
            .calculate_epoch_fee(get_recent_epoch()?, pre_fee_amount)
            .unwrap()
    } else {
        0
    };
    Ok(fee)
}

pub fn is_supported_mint(mint_account: &InterfaceAccount<Mint>) -> Result<bool> {
    let mint_info = mint_account.to_account_info();
    if *mint_info.owner == Token::id() {
        return Ok(true);
    }
    let mint_whitelist: HashSet<&str> = MINT_WHITELIST.into_iter().collect();
    if mint_whitelist.contains(mint_account.key().to_string().as_str()) {
        return Ok(true);
    }
    let mint_data = mint_info.try_borrow_data()?;
    let mint = StateWithExtensions::<spl_token_2022::state::Mint>::unpack(&mint_data)?;
    let extensions = mint.get_extension_types()?;
    for e in extensions {
        if e != ExtensionType::TransferFeeConfig
            && e != ExtensionType::MetadataPointer
            && e != ExtensionType::TokenMetadata
        {
            return Ok(false);
        }
    }
    Ok(true)
}
