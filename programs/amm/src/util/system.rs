use anchor_lang::{prelude::*, system_program};
use anchor_lang::solana_program::{program::invoke_signed, system_instruction};

pub fn create_or_allocate_account<'a>(
    program_id: &Pubkey,
    payer: AccountInfo<'a>,
    system_program: AccountInfo<'a>,
    target_account: AccountInfo<'a>,
    siger_seed: &[&[u8]],
    space: usize,
) -> Result<()> {
    let rent = Rent::get()?;
    let current_lamports = target_account.lamports();
    let signer_seeds = &[siger_seed];

    if current_lamports == 0 {
        let lamports = rent.minimum_balance(space);
        let ix = system_instruction::create_account(
            payer.key,
            target_account.key,
            lamports,
            u64::try_from(space).unwrap(),
            program_id,
        );
        invoke_signed(&ix, &[payer, target_account, system_program], signer_seeds)?;
    } else {
        let required_lamports = rent
            .minimum_balance(space)
            .max(1)
            .saturating_sub(current_lamports);
        if required_lamports > 0 {
            let ix = system_instruction::transfer(payer.key, target_account.key, required_lamports);
            invoke_signed(
                &ix,
                &[payer.clone(), target_account.clone(), system_program.clone()],
                &[],
            )?;
        }
        let ix = system_instruction::allocate(target_account.key, u64::try_from(space).unwrap());
        invoke_signed(
            &ix,
            &[target_account.clone(), system_program.clone()],
            signer_seeds,
        )?;

        let ix = system_instruction::assign(target_account.key, program_id);
        invoke_signed(&ix, &[target_account, system_program], signer_seeds)?;
    }
    Ok(())
}

#[cfg(not(test))]
pub fn get_recent_epoch() -> Result<u64> {
    Ok(Clock::get()?.epoch)
}

#[cfg(test)]
pub fn get_recent_epoch() -> Result<u64> {
    use std::time::{SystemTime, UNIX_EPOCH};
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        / (2 * 24 * 3600))
}
