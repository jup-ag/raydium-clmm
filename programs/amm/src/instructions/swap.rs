use crate::error::ErrorCode;
use crate::libraries::{
    big_num::U128, fixed_point_64, full_math::MulDiv, liquidity_math, swap_math, tick_math,
};
use crate::states::*;
use crate::util::*;
use anchor_lang::prelude::*;
use anchor_spl::token::Token;
use anchor_spl::token_interface::TokenAccount;
use std::cell::RefMut;
use std::collections::{HashMap, VecDeque};
#[cfg(feature = "enable-log")]
use std::convert::identity;
use std::ops::{Deref, Neg};

#[derive(Accounts)]
pub struct SwapSingle<'info> {
    /// The user performing the swap
    pub payer: Signer<'info>,

    /// The factory state to read protocol fees
    #[account(address = pool_state.load()?.amm_config)]
    pub amm_config: Box<Account<'info, AmmConfig>>,

    /// The program account of the pool in which the swap will be performed
    #[account(mut)]
    pub pool_state: AccountLoader<'info, PoolState>,

    /// The user token account for input token
    #[account(
        mut,
        token::token_program = token_program,
    )]
    pub input_token_account: Box<InterfaceAccount<'info, TokenAccount>>,

    /// The user token account for output token
    #[account(
        mut,
        token::token_program = token_program,
    )]
    pub output_token_account: Box<InterfaceAccount<'info, TokenAccount>>,

    /// The vault token account for input token
    #[account(
        mut,
        token::token_program = token_program,
    )]
    pub input_vault: Box<InterfaceAccount<'info, TokenAccount>>,

    /// The vault token account for output token
    #[account(
        mut,
        token::token_program = token_program,
    )]
    pub output_vault: Box<InterfaceAccount<'info, TokenAccount>>,

    /// The program account for the most recent oracle observation
    #[account(mut, address = pool_state.load()?.observation_key)]
    pub observation_state: AccountLoader<'info, ObservationState>,

    /// SPL program for token transfers
    pub token_program: Program<'info, Token>,

    #[account(mut, constraint = tick_array.load()?.pool_id == pool_state.key())]
    pub tick_array: AccountLoader<'info, TickArrayState>,
}

pub struct SwapAccounts<'b, 'info> {
    /// The user performing the swap
    pub signer: Signer<'info>,

    /// The user token account for input token
    pub input_token_account: Box<InterfaceAccount<'info, TokenAccount>>,

    /// The user token account for output token
    pub output_token_account: Box<InterfaceAccount<'info, TokenAccount>>,

    /// The vault token account for input token
    pub input_vault: Box<InterfaceAccount<'info, TokenAccount>>,

    /// The vault token account for output token
    pub output_vault: Box<InterfaceAccount<'info, TokenAccount>>,

    /// SPL program for token transfers
    pub token_program: Program<'info, Token>,

    /// The factory state to read protocol fees
    pub amm_config: &'b Box<Account<'info, AmmConfig>>,

    /// The program account of the pool in which the swap will be performed
    pub pool_state: &'b mut AccountLoader<'info, PoolState>,

    /// The tick_array account of current or next initialized
    pub tick_array_state: &'b mut AccountLoader<'info, TickArrayState>,

    /// The program account for the oracle observation
    pub observation_state: &'b mut AccountLoader<'info, ObservationState>,
}

/// Aggregated outcome returned by `swap_internal` after the swap loop finishes.
#[derive(Debug, Clone, Copy)]
pub struct SwapInternalResult {
    /// Net token_0 amount the pool absorbs (input side) or releases (output side)
    pub amount_0: u64,
    /// Net token_1 amount the pool absorbs (input side) or releases (output side)
    pub amount_1: u64,
    /// Total AMM trade fee (lp + protocol + fund) charged in token_0 during the swap
    pub trade_fee_0: u64,
    /// Total AMM trade fee (lp + protocol + fund) charged in token_1 during the swap
    pub trade_fee_1: u64,
    /// Pool sqrt(price) (Q64.64) after the swap
    pub sqrt_price_x64: u128,
    /// Pool liquidity after the swap
    pub liquidity: u128,
    /// Pool current tick after the swap
    pub tick: i32,
}

// the top level state of the swap, the results of which are recorded in storage at the end
#[derive(Debug)]
pub struct SwapState {
    // the amount remaining to be swapped in/out of the input/output asset
    pub amount_specified_remaining: u64,
    // the amount already swapped out/in of the output/input asset
    pub amount_calculated: u64,
    // The latest sqrt price of the pool
    pub sqrt_price_x64: u128,
    // the tick associated with the current price
    pub tick: i32,
    // the global fee growth of the token that receives fees this swap (depends on fee_on and zero_for_one)
    pub fee_growth_global_x64: u128,
    // the amount of input token paid as lp fee
    pub lp_fee: u64,
    // the amount of input token paid as protocol fee
    pub protocol_fee: u64,
    // the amount of input token paid as fund fee
    pub fund_fee: u64,
    // the current liquidity in range
    pub liquidity: u128,

    // the sqrt price for the next tick
    pub sqrt_price_next_x64: u128,
    // the next tick to swap to from the current tick in the swap direction
    pub tick_next: i32,

    // The tick spacing of the pool, used to group ticks for dynamic fee calculation
    pub tick_spacing: u16,
    // The base fee rate (static component) of the pool
    pub base_fee_rate: u32,
    // The current tick spacing index, representing which tick group the current price belongs to.
    // This is used to track volatility for dynamic fee calculation.
    pub tick_spacing_index: i32,
    // Dynamic fee configuration and state, including volatility accumulator and reference data.
    // None if dynamic fee is not enabled for this pool.
    pub dynamic_fee_info: Option<DynamicFeeInfo>,
}

impl SwapState {
    pub fn new(
        pool_state: &PoolState,
        amount_specified: u64,
        base_fee_rate: u32,
        zero_for_one: bool,
        block_timestamp: u64,
    ) -> Result<Self> {
        let mut state = Self {
            amount_specified_remaining: amount_specified,
            amount_calculated: 0,
            sqrt_price_x64: pool_state.sqrt_price_x64,
            tick: pool_state.tick_current,
            fee_growth_global_x64: if pool_state.is_fee_on_token0(zero_for_one) {
                pool_state.fee_growth_global_0_x64
            } else {
                pool_state.fee_growth_global_1_x64
            },
            lp_fee: 0,
            protocol_fee: 0,
            fund_fee: 0,
            liquidity: pool_state.liquidity,
            sqrt_price_next_x64: 0,
            tick_next: 0,
            base_fee_rate: base_fee_rate,
            tick_spacing: pool_state.tick_spacing,
            tick_spacing_index: 0,
            dynamic_fee_info: pool_state.get_dynamic_fee_info(),
        };
        if let Some(dynamic_fee_info) = &mut state.dynamic_fee_info {
            state.tick_spacing_index = tick_spacing_index_from_tick(state.tick, state.tick_spacing);
            dynamic_fee_info.update_reference(state.tick_spacing_index, block_timestamp)?;
        }
        Ok(state)
    }

    /// Apply swap step result by updating remaining and calculated amounts
    pub fn apply_swap_amounts(
        &mut self,
        amount_in: u64,
        amount_out: u64,
        fee_amount: u64,
        is_base_input: bool,
        is_fee_on_input: bool,
        protocol_fee_rate: u32,
        fund_fee_rate: u32,
    ) -> Result<()> {
        // Calculate the actual amount_in consumed by user
        // If fee is from input, user pays amount_in + fee_amount; otherwise just amount_in
        let amount_in_consumed = if is_fee_on_input {
            amount_in
                .checked_add(fee_amount)
                .ok_or(ErrorCode::CalculateOverflow)?
        } else {
            amount_in
        };

        if is_base_input {
            // Exact Input Swap: deduct consumed input from remaining, add output to calculated
            self.amount_specified_remaining = self
                .amount_specified_remaining
                .checked_sub(amount_in_consumed)
                .ok_or(ErrorCode::CalculateOverflow)?;
            // amount_out is already net output (fee already deducted if fee is from output)
            self.amount_calculated = self
                .amount_calculated
                .checked_add(amount_out)
                .ok_or(ErrorCode::CalculateOverflow)?;
        } else {
            // Exact Output Swap: deduct output from remaining, add consumed input to calculated
            self.amount_specified_remaining = self
                .amount_specified_remaining
                .checked_sub(amount_out)
                .ok_or(ErrorCode::CalculateOverflow)?;
            self.amount_calculated = self
                .amount_calculated
                .checked_add(amount_in_consumed)
                .ok_or(ErrorCode::CalculateOverflow)?;
        }
        self.spilt_fees(fee_amount, protocol_fee_rate, fund_fee_rate)?;
        Ok(())
    }

    pub fn spilt_fees(
        &mut self,
        fee_amont: u64,
        protocol_fee_rate: u32,
        fund_fee_rate: u32,
    ) -> Result<()> {
        let mut remaining_fee = fee_amont;
        // Process protocol fee
        if protocol_fee_rate > 0 {
            let protocol_fee_delta = U128::from(fee_amont)
                .checked_mul(protocol_fee_rate.into())
                .and_then(|v| v.checked_div(FEE_RATE_DENOMINATOR_VALUE.into()))
                .ok_or(ErrorCode::CalculateOverflow)?
                .as_u64();
            self.protocol_fee = self
                .protocol_fee
                .checked_add(protocol_fee_delta)
                .ok_or(ErrorCode::CalculateOverflow)?;
            remaining_fee = remaining_fee
                .checked_sub(protocol_fee_delta)
                .ok_or(ErrorCode::CalculateOverflow)?;
        }

        // Process fund fee
        if fund_fee_rate > 0 {
            let fund_fee_delta = U128::from(fee_amont)
                .checked_mul(fund_fee_rate.into())
                .and_then(|v| v.checked_div(FEE_RATE_DENOMINATOR_VALUE.into()))
                .ok_or(ErrorCode::CalculateOverflow)?
                .as_u64();

            self.fund_fee = self
                .fund_fee
                .checked_add(fund_fee_delta)
                .ok_or(ErrorCode::CalculateOverflow)?;
            remaining_fee = remaining_fee
                .checked_sub(fund_fee_delta)
                .ok_or(ErrorCode::CalculateOverflow)?;
        }

        // Update global fee tracker
        if self.liquidity > 0 {
            let fee_growth_global_x64_delta = U128::from(remaining_fee)
                .mul_div_floor(U128::from(fixed_point_64::Q64), U128::from(self.liquidity))
                .ok_or(ErrorCode::CalculateOverflow)?
                .as_u128();

            self.fee_growth_global_x64 = self
                .fee_growth_global_x64
                .wrapping_add(fee_growth_global_x64_delta);
            self.lp_fee = self
                .lp_fee
                .checked_add(remaining_fee)
                .ok_or(ErrorCode::CalculateOverflow)?;
        } else {
            self.protocol_fee = self
                .protocol_fee
                .checked_add(remaining_fee)
                .ok_or(ErrorCode::CalculateOverflow)?;
        }
        Ok(())
    }

    /// Settle the per-token deltas and trade-fee split after the swap loop.
    /// Returns `(amount_0, amount_1, trade_fee_0, trade_fee_1)`:
    /// - `amount_*` are the net token_0 / token_1 deltas the pool absorbs or releases.
    /// - `trade_fee_*` is the total AMM trade fee (lp + protocol + fund) charged on the
    ///   fee-side token (the other is zero).
    pub fn settle_amounts(
        &self,
        amount_specified: u64,
        zero_for_one: bool,
        is_base_input: bool,
        fee_on_token0: bool,
    ) -> Result<(u64, u64, u64, u64)> {
        let consumed = amount_specified
            .checked_sub(self.amount_specified_remaining)
            .ok_or(ErrorCode::CalculateOverflow)?;
        let (amount_0, amount_1) = if zero_for_one == is_base_input {
            (consumed, self.amount_calculated)
        } else {
            (self.amount_calculated, consumed)
        };

        let total_trade_fee = self
            .lp_fee
            .checked_add(self.protocol_fee)
            .and_then(|v| v.checked_add(self.fund_fee))
            .ok_or(ErrorCode::CalculateOverflow)?;
        let (trade_fee_0, trade_fee_1) = if fee_on_token0 {
            (total_trade_fee, 0u64)
        } else {
            (0u64, total_trade_fee)
        };

        Ok((amount_0, amount_1, trade_fee_0, trade_fee_1))
    }

    pub fn get_target_price_based_on_next_tick(
        &mut self,
        tick_next: i32,
        zero_for_one: bool,
        sqrt_price_limit_x64: u128,
    ) -> Result<u128> {
        // Clamp tick_next to valid range
        self.tick_next = tick_next;
        if self.tick_next < tick_math::MIN_TICK {
            self.tick_next = tick_math::MIN_TICK;
        } else if self.tick_next > tick_math::MAX_TICK {
            self.tick_next = tick_math::MAX_TICK;
        }

        // Calculate sqrt_price for the next tick
        self.sqrt_price_next_x64 = tick_math::get_sqrt_price_at_tick(self.tick_next)?;

        // Determine target price: either the next tick price or the limit price
        let target_price = if (zero_for_one && self.sqrt_price_next_x64 < sqrt_price_limit_x64)
            || (!zero_for_one && self.sqrt_price_next_x64 > sqrt_price_limit_x64)
        {
            sqrt_price_limit_x64
        } else {
            self.sqrt_price_next_x64
        };

        // Validate swap direction
        if zero_for_one {
            require_gte!(self.tick, self.tick_next);
            require_gte!(self.sqrt_price_x64, self.sqrt_price_next_x64);
            require_gte!(self.sqrt_price_x64, target_price);
        } else {
            require_gt!(self.tick_next, self.tick);
            require_gte!(self.sqrt_price_next_x64, self.sqrt_price_x64);
            require_gte!(target_price, self.sqrt_price_x64);
        }

        Ok(target_price)
    }

    pub fn update_volatility_accumulator(&mut self) -> Result<()> {
        if let Some(dynamic_fee_info) = &mut self.dynamic_fee_info {
            dynamic_fee_info.update_volatility_accumulator(self.tick_spacing_index)?;
        }
        Ok(())
    }

    pub fn update_dynamic_fee_index(
        &mut self,
        zero_for_one: bool,
        is_skipped_tick_spacing: bool,
    ) -> Result<()> {
        if let Some(dynamic_fee_info) = &self.dynamic_fee_info {
            if is_skipped_tick_spacing {
                let tick_index = if self.sqrt_price_x64 == self.sqrt_price_next_x64 {
                    self.tick_next
                } else {
                    self.tick
                };
                let mut tick_spacing_index =
                    tick_spacing_index_from_tick(tick_index, self.tick_spacing);
                if !zero_for_one && tick_index % (self.tick_spacing as i32) == 0 {
                    tick_spacing_index = tick_spacing_index - 1;
                }
                self.tick_spacing_index = tick_spacing_index;

                if dynamic_fee_info.volatility_accumulator
                    != dynamic_fee_info.max_volatility_accumulator
                {
                    self.update_volatility_accumulator()?;
                }
            }
            self.tick_spacing_index += if zero_for_one { -1 } else { 1 };
        }
        Ok(())
    }

    pub fn update_volatility_accumulator_on_price(&mut self) -> Result<()> {
        if self.dynamic_fee_info.is_some() {
            let tick_index = tick_math::get_tick_at_sqrt_price(self.sqrt_price_x64)?;
            let final_tick_spacing_index =
                tick_spacing_index_from_tick(tick_index, self.tick_spacing);
            if self.tick_spacing_index != final_tick_spacing_index {
                self.tick_spacing_index = final_tick_spacing_index;
                self.update_volatility_accumulator()?;
            }
        }
        Ok(())
    }

    pub fn get_spacing_bounded_price(
        &self,
        target_price: u128,
        zero_for_one: bool,
    ) -> Result<(bool, u128)> {
        if let Some(dynamic_fee_info) = &self.dynamic_fee_info {
            if self.liquidity == 0
                || dynamic_fee_info.volatility_accumulator
                    == dynamic_fee_info.max_volatility_accumulator
            {
                return Ok((true, target_price));
            }

            let tick_spacing_i32 = i32::from(self.tick_spacing);
            let bounded_tick = if zero_for_one {
                self.tick_spacing_index.saturating_mul(tick_spacing_i32)
            } else {
                self.tick_spacing_index
                    .saturating_add(1)
                    .saturating_mul(tick_spacing_i32)
            };
            #[cfg(feature = "enable-log")]
            msg!(
                "state.tick:{}, state.tick_spacing_index:{}, bounded_tick:{}",
                self.tick,
                self.tick_spacing_index,
                bounded_tick
            );
            let bounded_sqrt_price = tick_math::get_sqrt_price_at_tick(
                bounded_tick.clamp(tick_math::MIN_TICK, tick_math::MAX_TICK),
            )?;

            if zero_for_one {
                Ok((false, target_price.max(bounded_sqrt_price)))
            } else {
                Ok((false, target_price.min(bounded_sqrt_price)))
            }
        } else {
            Ok((true, target_price))
        }
    }

    pub fn get_total_fee_rate(&self) -> Result<u32> {
        // Use base + dynamic fee if dynamic fee is enabled
        if let Some(dynamic_fee_info) = &self.dynamic_fee_info {
            let dynamic_fee_rate =
                Self::compute_dynamic_fee_rate(dynamic_fee_info, self.tick_spacing)?;
            let total_fee_rate = self.base_fee_rate + dynamic_fee_rate;
            return Ok(total_fee_rate.min(MAX_FEE_RATE_NUMERATOR));
        }
        // Use base fee if not in launch phase and dynamic fee is disabled
        Ok(self.base_fee_rate)
    }

    /// Computes the dynamic fee rate based on volatility accumulator.
    ///
    /// The dynamic fee rate is calculated using a quadratic formula that maps the squared
    /// volatility accumulator to a fee rate. This creates a non-linear relationship where
    /// higher volatility results in exponentially higher fees, providing stronger protection
    /// against market manipulation during volatile periods.
    fn compute_dynamic_fee_rate(
        dynamic_fee_info: &DynamicFeeInfo,
        tick_spacing: u16,
    ) -> Result<u32> {
        let crossed = dynamic_fee_info.volatility_accumulator * tick_spacing as u32;

        // Square the crossed value to create quadratic fee scaling
        let squared = u64::from(crossed) * u64::from(crossed);

        let denominator = U128::from(DYNAMIC_FEE_CONTROL_DENOMINATOR)
            * U128::from(VOLATILITY_ACCUMULATOR_SCALE)
            * U128::from(VOLATILITY_ACCUMULATOR_SCALE);

        // Compute fee rate using ceiling division to ensure minimum fee protection
        let fee_rate = U128::from(dynamic_fee_info.dynamic_fee_control)
            .mul_div_ceil(U128::from(squared), denominator)
            .ok_or(ErrorCode::CalculateOverflow)?
            .as_u128();
        // bound the fee rate to the maximum fee rate
        if fee_rate > MAX_FEE_RATE_NUMERATOR as u128 {
            Ok(MAX_FEE_RATE_NUMERATOR)
        } else {
            Ok(fee_rate as u32)
        }
    }
}

#[derive(Default)]
struct StepComputations {
    // the price at the beginning of the step
    sqrt_price_start_x64: u128,
    // the next tick to swap to from the current tick in the swap direction
    tick_next: i32,
    // whether tick_next is initialized or not
    initialized: bool,
    // sqrt(price) for the next tick (1/0)
    sqrt_price_next_x64: u128,
    // how much is being swapped in in this step
    amount_in: u64,
    // how much is being swapped out
    amount_out: u64,
    // how much fee is being paid in
    fee_amount: u64,
}

pub fn swap_internal<'b, 'c: 'info, 'info>(
    amm_config: &AmmConfig,
    pool_state: &mut RefMut<PoolState>,
    tick_array_states: &mut VecDeque<RefMut<TickArrayState>>,
    observation_state: &mut RefMut<ObservationState>,
    tickarray_bitmap_extension_info: Option<&'c AccountInfo<'info>>,
    amount_specified: u64,
    sqrt_price_limit_x64: u128,
    zero_for_one: bool,
    is_base_input: bool,
    block_timestamp: u32,
) -> Result<SwapInternalResult> {
    require!(amount_specified != 0, ErrorCode::ZeroAmountSpecified);
    if !pool_state.get_status_by_bit(PoolStatusBitIndex::Swap) {
        return err!(ErrorCode::NotApproved);
    }
    require!(
        if zero_for_one {
            sqrt_price_limit_x64 < pool_state.sqrt_price_x64
                && sqrt_price_limit_x64 > tick_math::MIN_SQRT_PRICE_X64
        } else {
            sqrt_price_limit_x64 > pool_state.sqrt_price_x64
                && sqrt_price_limit_x64 < tick_math::MAX_SQRT_PRICE_X64
        },
        ErrorCode::SqrtPriceLimitOverflow
    );

    let updated_reward_infos = pool_state.update_reward_infos(block_timestamp as u64)?;
    // check observation account is owned by the pool
    require_keys_eq!(observation_state.pool_id, pool_state.key());

    let (mut first_tick_array_contains_pool_tick, first_valid_tick_array_start_index) = pool_state
        .first_tick_array_index_with_extension_info(
            tickarray_bitmap_extension_info,
            zero_for_one,
        )?;
    let mut current_valid_tick_array_start_index = first_valid_tick_array_start_index;

    let mut tick_array_current = tick_array_states
        .pop_front()
        .ok_or(ErrorCode::NotEnoughTickArrayAccount)?;
    // find the first active tick array account
    for _ in 0..tick_array_states.len() {
        if tick_array_current.start_tick_index == current_valid_tick_array_start_index {
            break;
        }
        tick_array_current = tick_array_states
            .pop_front()
            .ok_or(ErrorCode::NotEnoughTickArrayAccount)?;
    }
    // check the first tick_array account is owned by the pool
    require_keys_eq!(tick_array_current.pool_id, pool_state.key());
    // check first tick array account is correct
    require_eq!(
        tick_array_current.start_tick_index,
        current_valid_tick_array_start_index,
        ErrorCode::InvalidFirstTickArrayAccount
    );

    // Determine if fee should be collected from input token (only need to calculate once)
    let is_fee_on_input = pool_state.is_fee_on_input(zero_for_one);
    let mut state = SwapState::new(
        pool_state,
        amount_specified,
        amm_config.trade_fee_rate,
        zero_for_one,
        block_timestamp as u64,
    )?;

    // Main swap loop: continue swapping until we've consumed all input/output or reached the price limit
    // Each iteration processes one step from current price to the next initialized tick
    while state.amount_specified_remaining != 0 && state.sqrt_price_x64 != sqrt_price_limit_x64 {
        #[cfg(feature = "enable-log")]
        msg!("begin, is_base_input:{}, state.liquidity:{}, state.tick:{}, state.sqrt_price_x64:{}, state.tick_spacing_index:{}", is_base_input, state.liquidity, state.tick, state.sqrt_price_x64, state.tick_spacing_index);

        let mut next_initialized_tick = {
            // First, try to find next initialized tick in current tick array
            if let Some(tick_state) = tick_array_current.next_initialized_tick(
                state.tick,
                pool_state.tick_spacing,
                zero_for_one,
            )? {
                *tick_state
            }
            // If not found and the first tick array doesn't contain pool's current tick,
            // use the first initialized tick in current array (only happens once in the first iteration)
            else if !first_tick_array_contains_pool_tick {
                first_tick_array_contains_pool_tick = true;
                *tick_array_current.first_initialized_tick(zero_for_one)?
            }
            // Otherwise, need to move to next tick array
            else {
                let next_tick_array_index = pool_state
                    .next_tick_array_index_with_extension_info(
                        tickarray_bitmap_extension_info,
                        current_valid_tick_array_start_index,
                        zero_for_one,
                    )?
                    .ok_or(ErrorCode::LiquidityInsufficient)?;

                // Advance to the next tick array
                while tick_array_current.start_tick_index != next_tick_array_index {
                    tick_array_current = tick_array_states
                        .pop_front()
                        .ok_or(ErrorCode::NotEnoughTickArrayAccount)?;
                    // check the tick_array account is owned by the pool
                    require_keys_eq!(tick_array_current.pool_id, pool_state.key());
                }
                current_valid_tick_array_start_index = next_tick_array_index;

                *tick_array_current.first_initialized_tick(zero_for_one)?
            }
        };
        #[cfg(feature = "enable-log")]
        msg!(
            "next_initialized_tick:{}, tick_array_current:{}",
            identity(next_initialized_tick.tick),
            tick_array_current.key().to_string(),
        );
        require_eq!(next_initialized_tick.is_initialized(), true);

        let target_price = state.get_target_price_based_on_next_tick(
            next_initialized_tick.tick,
            zero_for_one,
            sqrt_price_limit_x64,
        )?;
        #[cfg(feature = "enable-log")]
        msg!(
            "state.tick_next:{}, state.sqrt_price_next_x64:{}",
            state.tick_next,
            state.sqrt_price_next_x64
        );

        let mut liquidity_next = state.liquidity;
        loop {
            state.update_volatility_accumulator()?;
            let total_fee_rate = state.get_total_fee_rate()?;
            let (is_skipped_tick_spacing, bounded_price) =
                state.get_spacing_bounded_price(target_price, zero_for_one)?;

            let is_price_change = state.sqrt_price_x64 != bounded_price;
            let swap_computed_result = if is_price_change {
                let swap_computed_result = swap_math::compute_swap(
                    state.sqrt_price_x64,
                    bounded_price,
                    state.liquidity,
                    state.amount_specified_remaining,
                    total_fee_rate,
                    is_base_input,
                    zero_for_one,
                    is_fee_on_input,
                )?;
                #[cfg(feature = "enable-log")]
                msg!(
                    "swap_computed_result: amount_in:{}, amount_out:{}, fee_amount:{}",
                    swap_computed_result.amount_in,
                    swap_computed_result.amount_out,
                    swap_computed_result.fee_amount
                );
                state.apply_swap_amounts(
                    swap_computed_result.amount_in,
                    swap_computed_result.amount_out,
                    swap_computed_result.fee_amount,
                    is_base_input,
                    is_fee_on_input,
                    amm_config.protocol_fee_rate,
                    amm_config.fund_fee_rate,
                )?;
                swap_computed_result
            } else {
                swap_math::SwapComputationResult::new(bounded_price)
            };
            let limit_order_unfilled_amount_before =
                next_initialized_tick.limit_order_unfilled_amount()?;
            if state.sqrt_price_next_x64 == swap_computed_result.sqrt_price_next_x64 {
                // try to match limit orders on this tick
                let limit_order_result = next_initialized_tick.match_limit_order(
                    state.amount_specified_remaining,
                    zero_for_one,
                    is_base_input,
                    total_fee_rate,
                    is_fee_on_input,
                )?;

                if limit_order_result.amount_in > 0 {
                    #[cfg(feature = "enable-log")]
                    msg!(
                        "limit_order_result: amount_in:{}, amount_out:{}, amm_fee_amount:{}",
                        limit_order_result.amount_in,
                        limit_order_result.amount_out,
                        limit_order_result.amm_fee_amount
                    );
                    state.apply_swap_amounts(
                        limit_order_result.amount_in,
                        limit_order_result.amount_out,
                        limit_order_result.amm_fee_amount,
                        is_base_input,
                        is_fee_on_input,
                        amm_config.protocol_fee_rate,
                        amm_config.fund_fee_rate,
                    )?;
                }

                if !next_initialized_tick.is_initialized() {
                    tick_array_current.update_initialized_tick_count(false)?;
                    if tick_array_current.initialized_tick_count == 0 {
                        pool_state.flip_tick_array_bit(
                            tickarray_bitmap_extension_info,
                            tick_array_current.start_tick_index,
                        )?;
                    }
                }

                if next_initialized_tick.has_liquidity()
                    && !next_initialized_tick.has_limit_orders()
                {
                    // Use current fee growth for each token: the one that receives fees this swap
                    // is updated in state, the other stays at pool value.
                    let fee_on_token0 = pool_state.is_fee_on_token0(zero_for_one);
                    let mut liquidity_net = next_initialized_tick.cross(
                        if fee_on_token0 {
                            state.fee_growth_global_x64
                        } else {
                            pool_state.fee_growth_global_0_x64
                        },
                        if fee_on_token0 {
                            pool_state.fee_growth_global_1_x64
                        } else {
                            state.fee_growth_global_x64
                        },
                        &updated_reward_infos,
                    );
                    if zero_for_one {
                        liquidity_net = liquidity_net.neg();
                    }
                    liquidity_next = liquidity_math::add_delta(state.liquidity, liquidity_net)?;
                }

                tick_array_current.update_tick_state(
                    next_initialized_tick.tick,
                    pool_state.tick_spacing.into(),
                    next_initialized_tick,
                )?;

                // Update tick based on limit order status and swap direction
                // The tick assignment rule:
                // - zero_for_one=true && has_limit_orders: tick = tick_next
                // - zero_for_one=false && has_limit_orders: tick = tick_next - 1
                // - zero_for_one=true && !has_limit_orders: tick = tick_next - 1
                // - zero_for_one=false && !has_limit_orders: tick = tick_next
                state.tick = if (zero_for_one && !next_initialized_tick.has_limit_orders())
                    || (!zero_for_one && next_initialized_tick.has_limit_orders())
                {
                    state.tick_next - 1
                } else {
                    state.tick_next
                };
            } else if state.sqrt_price_x64 != swap_computed_result.sqrt_price_next_x64 {
                // recompute unless we're on a lower tick boundary (i.e. already transitioned ticks), and haven't moved
                // if only a small amount of quantity is traded, the input may be consumed by fees, resulting in no price change. If state.sqrt_price_x64, i.e., the latest price in the pool, is used to recalculate the tick, some errors may occur.
                // for example, if zero_for_one, and the price falls exactly on an initialized tick t after the first trade, then at this point, pool.sqrtPriceX64 = get_sqrt_price_at_tick(t), while pool.tick = t-1. if the input quantity of the
                // second trade is very small and the pool price does not change after the transaction, if the tick is recalculated, pool.tick will be equal to t, which is incorrect.
                state.tick =
                    tick_math::get_tick_at_sqrt_price(swap_computed_result.sqrt_price_next_x64)?;
            }
            state.sqrt_price_x64 = swap_computed_result.sqrt_price_next_x64;
            if state.amount_specified_remaining == 0 || state.sqrt_price_x64 == target_price {
                let limit_order_unfilled_amount_after =
                    next_initialized_tick.limit_order_unfilled_amount()?;
                // One of the two parties must be equal to 0 to exit the loop
                // If a limit order has been executed and the active swap amount is not zero, then the remaining amount of the limit order must be zero, otherwise it's an abnormal situation
                if state.amount_specified_remaining != 0
                    && limit_order_unfilled_amount_after != limit_order_unfilled_amount_before
                {
                    require_eq!(limit_order_unfilled_amount_after, 0);
                }
                break;
            }
            state.update_dynamic_fee_index(zero_for_one, is_skipped_tick_spacing)?;
        }
        state.liquidity = liquidity_next;
    }
    // At the end of the entire swap loop, `update_dynamic_fee_index` does not always guarantee that
    // the tick_spacing_index lands in the correct position. Therefore, we recalculate its position
    // here based on the current price and update the volatility accumulator.
    state.update_volatility_accumulator_on_price()?;

    #[cfg(feature = "enable-log")]
    msg!("end, state:{:#?}", state);

    // Update pool state with final swap results
    if state.tick != pool_state.tick_current {
        // Update observation with previous tick before updating current tick
        observation_state.update(block_timestamp, pool_state.tick_current);
    }

    let (amount_0, amount_1, trade_fee_0, trade_fee_1) = state.settle_amounts(
        amount_specified,
        zero_for_one,
        is_base_input,
        pool_state.is_fee_on_token0(zero_for_one),
    )?;

    pool_state.update_after_swap(
        state.tick,
        state.sqrt_price_x64,
        state.liquidity,
        state.lp_fee,
        state.protocol_fee,
        state.fund_fee,
        state.fee_growth_global_x64,
        zero_for_one,
        state.dynamic_fee_info,
    )?;
    Ok(SwapInternalResult {
        amount_0,
        amount_1,
        trade_fee_0,
        trade_fee_1,
        sqrt_price_x64: pool_state.sqrt_price_x64,
        liquidity: pool_state.liquidity,
        tick: pool_state.tick_current,
    })
}

/// Performs a single exact input/output swap
/// if is_base_input = true, return vaule is the max_amount_out, otherwise is min_amount_in
pub fn exact_internal<'b, 'c: 'info, 'info>(
    ctx: &mut SwapAccounts<'b, 'info>,
    remaining_accounts: &'c [AccountInfo<'info>],
    amount_specified: u64,
    sqrt_price_limit_x64: u128,
    is_base_input: bool,
) -> Result<u64> {
    let block_timestamp = solana_program::clock::Clock::get()?.unix_timestamp as u64;

    let swap_result: SwapInternalResult;
    let zero_for_one;
    let swap_price_before;

    let input_balance_before = ctx.input_vault.amount;
    let output_balance_before = ctx.output_vault.amount;

    {
        swap_price_before = ctx.pool_state.load()?.sqrt_price_x64;
        let pool_state = &mut ctx.pool_state.load_mut()?;
        zero_for_one = ctx.input_vault.mint == pool_state.token_mint_0;

        require_gt!(block_timestamp, pool_state.open_time);

        require!(
            if zero_for_one {
                ctx.input_vault.key() == pool_state.token_vault_0
                    && ctx.output_vault.key() == pool_state.token_vault_1
            } else {
                ctx.input_vault.key() == pool_state.token_vault_1
                    && ctx.output_vault.key() == pool_state.token_vault_0
            },
            ErrorCode::InvalidInputPoolVault
        );

        let mut tickarray_bitmap_extension = None;
        let tick_array_states = &mut VecDeque::new();
        tick_array_states.push_back(ctx.tick_array_state.load_mut()?);

        let tick_array_bitmap_extension_key = TickArrayBitmapExtension::key(pool_state.key());
        for account_info in remaining_accounts.into_iter() {
            if account_info.key().eq(&tick_array_bitmap_extension_key) {
                tickarray_bitmap_extension = Some(account_info);
                continue;
            }
            tick_array_states.push_back(AccountLoad::load_data_mut(account_info)?);
        }

        swap_result = swap_internal(
            &ctx.amm_config,
            pool_state,
            tick_array_states,
            &mut ctx.observation_state.load_mut()?,
            tickarray_bitmap_extension,
            amount_specified,
            if sqrt_price_limit_x64 == 0 {
                if zero_for_one {
                    tick_math::MIN_SQRT_PRICE_X64 + 1
                } else {
                    tick_math::MAX_SQRT_PRICE_X64 - 1
                }
            } else {
                sqrt_price_limit_x64
            },
            zero_for_one,
            is_base_input,
            oracle::block_timestamp(),
        )?;

        #[cfg(feature = "enable-log")]
        msg!(
            "exact_swap_internal, is_base_input:{}, amount_0: {}, amount_1: {}",
            is_base_input,
            swap_result.amount_0,
            swap_result.amount_1
        );
        require!(
            swap_result.amount_0 != 0 && swap_result.amount_1 != 0,
            ErrorCode::TooSmallInputOrOutputAmount
        );
    }
    let (token_account_0, token_account_1, vault_0, vault_1) = if zero_for_one {
        (
            ctx.input_token_account.clone(),
            ctx.output_token_account.clone(),
            ctx.input_vault.clone(),
            ctx.output_vault.clone(),
        )
    } else {
        (
            ctx.output_token_account.clone(),
            ctx.input_token_account.clone(),
            ctx.output_vault.clone(),
            ctx.input_vault.clone(),
        )
    };

    if zero_for_one {
        //  x -> y, deposit x token from user to pool vault.
        transfer_from_user_to_pool_vault(
            &ctx.signer,
            &token_account_0,
            &vault_0,
            None,
            &ctx.token_program,
            None,
            swap_result.amount_0,
        )?;
        if vault_1.amount <= swap_result.amount_1 {
            // freeze pool, disable all instructions
            ctx.pool_state.load_mut()?.set_status(255);
        }
        // x -> y，transfer y token from pool vault to user.
        transfer_from_pool_vault_to_user(
            &ctx.pool_state,
            &vault_1,
            &token_account_1,
            None,
            &ctx.token_program,
            None,
            swap_result.amount_1,
        )?;
    } else {
        transfer_from_user_to_pool_vault(
            &ctx.signer,
            &token_account_1,
            &vault_1,
            None,
            &ctx.token_program,
            None,
            swap_result.amount_1,
        )?;
        if vault_0.amount <= swap_result.amount_0 {
            // freeze pool, disable all instructions
            ctx.pool_state.load_mut()?.set_status(255);
        }
        transfer_from_pool_vault_to_user(
            &ctx.pool_state,
            &vault_0,
            &token_account_0,
            None,
            &ctx.token_program,
            None,
            swap_result.amount_0,
        )?;
    }
    ctx.output_vault.reload()?;
    ctx.input_vault.reload()?;

    emit!(SwapEvent {
        pool_state: ctx.pool_state.key(),
        sender: ctx.signer.key(),
        token_account_0: token_account_0.key(),
        token_account_1: token_account_1.key(),
        amount_0: swap_result.amount_0,
        transfer_fee_0: 0,
        amount_1: swap_result.amount_1,
        transfer_fee_1: 0,
        zero_for_one,
        sqrt_price_x64: swap_result.sqrt_price_x64,
        liquidity: swap_result.liquidity,
        tick: swap_result.tick,
        trade_fee_0: swap_result.trade_fee_0,
        trade_fee_1: swap_result.trade_fee_1,
    });
    if zero_for_one {
        require_gt!(swap_price_before, swap_result.sqrt_price_x64);
    } else {
        require_gt!(swap_result.sqrt_price_x64, swap_price_before);
    }
    if sqrt_price_limit_x64 == 0 {
        // Does't allow partial filled without specified limit_price.
        if is_base_input {
            if zero_for_one {
                require_eq!(amount_specified, swap_result.amount_0);
            } else {
                require_eq!(amount_specified, swap_result.amount_1);
            }
        } else {
            if zero_for_one {
                require_eq!(amount_specified, swap_result.amount_1);
            } else {
                require_eq!(amount_specified, swap_result.amount_0);
            }
        }
    }

    if is_base_input {
        Ok(output_balance_before
            .checked_sub(ctx.output_vault.amount)
            .unwrap())
    } else {
        Ok(ctx
            .input_vault
            .amount
            .checked_sub(input_balance_before)
            .unwrap())
    }
}

pub fn swap<'info>(
    ctx: Context<'info, SwapSingle<'info>>,
    amount: u64,
    other_amount_threshold: u64,
    sqrt_price_limit_x64: u128,
    is_base_input: bool,
) -> Result<()> {
    let amount = exact_internal(
        &mut SwapAccounts {
            signer: ctx.accounts.payer.clone(),
            amm_config: &ctx.accounts.amm_config,
            input_token_account: ctx.accounts.input_token_account.clone(),
            output_token_account: ctx.accounts.output_token_account.clone(),
            input_vault: ctx.accounts.input_vault.clone(),
            output_vault: ctx.accounts.output_vault.clone(),
            token_program: ctx.accounts.token_program.clone(),
            pool_state: &mut ctx.accounts.pool_state,
            tick_array_state: &mut ctx.accounts.tick_array,
            observation_state: &mut ctx.accounts.observation_state,
        },
        ctx.remaining_accounts,
        amount,
        sqrt_price_limit_x64,
        is_base_input,
    )?;
    if is_base_input {
        require!(
            amount >= other_amount_threshold,
            ErrorCode::TooLittleOutputReceived
        );
    } else {
        require!(
            amount <= other_amount_threshold,
            ErrorCode::TooMuchInputPaid
        );
    }

    Ok(())
}

/// Off-chain quote-path mirror of `swap_internal`. Same swap math, but doesn't mutate
/// `pool_state`, observation state, or tick array accounts — operates on references and
/// returns the final `SwapState` plus `(amount_0, amount_1)`. Limit-order fills, dynamic fee,
/// and `fee_on` are all handled identically to the on-chain version.
pub fn swap_on_swap_state(
    amm_config: &AmmConfig,
    pool_state: &PoolState,
    tick_array_states: VecDeque<&TickArrayState>,
    tickarray_bitmap_extension: &Option<TickArrayBitmapExtension>,
    amount_specified: u64,
    sqrt_price_limit_x64: u128,
    zero_for_one: bool,
    is_base_input: bool,
    block_timestamp: u64,
) -> Result<(SwapState, u64, u64)> {
    swap_on_swap_state_with_cache(
        amm_config,
        pool_state,
        tick_array_states,
        tickarray_bitmap_extension,
        amount_specified,
        sqrt_price_limit_x64,
        zero_for_one,
        is_base_input,
        block_timestamp,
        None,
    )
}

#[derive(Default)]
pub struct SwapQuoteCache {
    tick_masks: HashMap<(Pubkey, i32), TickArrayMaskCache>,
}

#[derive(Default, Clone, Copy)]
struct TickArrayMaskCache {
    mask: u64,
    recent_epoch: u64,
}

impl SwapQuoteCache {
    #[inline(always)]
    fn mask_for(&mut self, tick_array: &TickArrayState) -> u64 {
        let key = (tick_array.pool_id, tick_array.start_tick_index);
        let entry = self
            .tick_masks
            .entry(key)
            .or_insert_with(|| TickArrayMaskCache {
                mask: build_initialized_mask(tick_array),
                recent_epoch: tick_array.recent_epoch,
            });

        if entry.recent_epoch != tick_array.recent_epoch {
            entry.mask = build_initialized_mask(tick_array);
            entry.recent_epoch = tick_array.recent_epoch;
        }

        entry.mask
    }
}

#[inline(always)]
fn build_initialized_mask(tick_array: &TickArrayState) -> u64 {
    if tick_array.initialized_tick_count == 0 {
        return 0;
    }
    let mut mask: u64 = 0;
    let mut i: usize = 0;
    while i < crate::states::tick_array::TICK_ARRAY_SIZE_USIZE {
        if tick_array.ticks[i].is_initialized() {
            mask |= 1u64 << i;
        }
        i += 1;
    }
    mask
}

/// Quote-path mirror of `swap_internal`. Side-by-side comparison invariants:
///   - Same swap-step math, dynamic-fee accounting, and limit-order fill path
///   - Quote-path adjustments (vs on-chain swap_internal):
///       * Takes `&PoolState` + `VecDeque<&TickArrayState>` instead of `RefMut`s
///       * Takes `&Option<TickArrayBitmapExtension>` (deserialized) instead of `AccountInfo`,
///         using the matching `get_first_initialized_tick_array` /
///         `next_initialized_tick_array_start_index` overloads
///       * No `observation_state` (no oracle update)
///       * No `pool_state.update_reward_infos` — `cross()` is called with `pool_state.reward_infos`
///         directly; the writes to the local `next_initialized_tick` copy are discarded anyway
///       * No `pool_state.update_after_swap` — pool state isn't mutated
///       * No `tick_array_current.update_initialized_tick_count` / `flip_tick_array_bit` /
///         `update_tick_state` — tick array state isn't mutated
///       * No `require_keys_eq!` / `require_eq!` checks against pool/account identity
pub fn swap_on_swap_state_with_cache(
    amm_config: &AmmConfig,
    pool_state: &PoolState,
    mut tick_array_states: VecDeque<&TickArrayState>,
    tickarray_bitmap_extension: &Option<TickArrayBitmapExtension>,
    amount_specified: u64,
    sqrt_price_limit_x64: u128,
    zero_for_one: bool,
    is_base_input: bool,
    block_timestamp: u64,
    mut _quote_cache: Option<&mut SwapQuoteCache>,
) -> Result<(SwapState, u64, u64)> {
    require!(amount_specified != 0, ErrorCode::ZeroAmountSpecified);
    if !pool_state.get_status_by_bit(PoolStatusBitIndex::Swap) {
        return err!(ErrorCode::NotApproved);
    }
    require!(
        if zero_for_one {
            sqrt_price_limit_x64 < pool_state.sqrt_price_x64
                && sqrt_price_limit_x64 > tick_math::MIN_SQRT_PRICE_X64
        } else {
            sqrt_price_limit_x64 > pool_state.sqrt_price_x64
                && sqrt_price_limit_x64 < tick_math::MAX_SQRT_PRICE_X64
        },
        ErrorCode::SqrtPriceLimitOverflow
    );

    // Quote-path: skip `pool_state.update_reward_infos` (mutating; not needed for quote math).
    // Quote-path: skip `require_keys_eq!(observation_state.pool_id, pool_state.key())`.

    let (mut first_tick_array_contains_pool_tick, first_valid_tick_array_start_index) =
        pool_state.get_first_initialized_tick_array(tickarray_bitmap_extension, zero_for_one)?;
    let mut current_valid_tick_array_start_index = first_valid_tick_array_start_index;

    let mut tick_array_current = tick_array_states
        .pop_front()
        .ok_or(ErrorCode::NotEnoughTickArrayAccount)?;
    for _ in 0..tick_array_states.len() {
        if tick_array_current.start_tick_index == current_valid_tick_array_start_index {
            break;
        }
        tick_array_current = tick_array_states
            .pop_front()
            .ok_or(ErrorCode::NotEnoughTickArrayAccount)?;
    }
    // Quote-path: skip `require_keys_eq!(tick_array_current.pool_id, pool_state.key())` and
    // `require_eq!(tick_array_current.start_tick_index, current_valid_tick_array_start_index, ...)`.

    let is_fee_on_input = pool_state.is_fee_on_input(zero_for_one);
    let mut state = SwapState::new(
        pool_state,
        amount_specified,
        amm_config.trade_fee_rate,
        zero_for_one,
        block_timestamp,
    )?;

    while state.amount_specified_remaining != 0 && state.sqrt_price_x64 != sqrt_price_limit_x64 {
        #[cfg(feature = "enable-log")]
        msg!("begin, is_base_input:{}, state.liquidity:{}, state.tick:{}, state.sqrt_price_x64:{}, state.tick_spacing_index:{}", is_base_input, state.liquidity, state.tick, state.sqrt_price_x64, state.tick_spacing_index);

        let mut next_initialized_tick = {
            if let Some(tick_state) = tick_array_current.next_initialized_tick(
                state.tick,
                pool_state.tick_spacing,
                zero_for_one,
            )? {
                *tick_state
            } else if !first_tick_array_contains_pool_tick {
                first_tick_array_contains_pool_tick = true;
                *tick_array_current.first_initialized_tick(zero_for_one)?
            } else {
                let next_tick_array_index = pool_state
                    .next_initialized_tick_array_start_index(
                        tickarray_bitmap_extension,
                        current_valid_tick_array_start_index,
                        zero_for_one,
                    )?
                    .ok_or(ErrorCode::LiquidityInsufficient)?;

                // Quote-path: skip `require_keys_eq!(tick_array_current.pool_id, pool_state.key())`.
                while tick_array_current.start_tick_index != next_tick_array_index {
                    tick_array_current = tick_array_states
                        .pop_front()
                        .ok_or(ErrorCode::NotEnoughTickArrayAccount)?;
                }
                current_valid_tick_array_start_index = next_tick_array_index;

                *tick_array_current.first_initialized_tick(zero_for_one)?
            }
        };
        #[cfg(feature = "enable-log")]
        msg!(
            "next_initialized_tick:{}",
            identity(next_initialized_tick.tick),
        );
        require_eq!(next_initialized_tick.is_initialized(), true);

        let target_price = state.get_target_price_based_on_next_tick(
            next_initialized_tick.tick,
            zero_for_one,
            sqrt_price_limit_x64,
        )?;

        let mut liquidity_next = state.liquidity;
        loop {
            state.update_volatility_accumulator()?;
            let total_fee_rate = state.get_total_fee_rate()?;
            let (is_skipped_tick_spacing, bounded_price) =
                state.get_spacing_bounded_price(target_price, zero_for_one)?;

            let is_price_change = state.sqrt_price_x64 != bounded_price;
            let swap_computed_result = if is_price_change {
                let swap_computed_result = swap_math::compute_swap(
                    state.sqrt_price_x64,
                    bounded_price,
                    state.liquidity,
                    state.amount_specified_remaining,
                    total_fee_rate,
                    is_base_input,
                    zero_for_one,
                    is_fee_on_input,
                )?;
                state.apply_swap_amounts(
                    swap_computed_result.amount_in,
                    swap_computed_result.amount_out,
                    swap_computed_result.fee_amount,
                    is_base_input,
                    is_fee_on_input,
                    amm_config.protocol_fee_rate,
                    amm_config.fund_fee_rate,
                )?;
                swap_computed_result
            } else {
                swap_math::SwapComputationResult::new(bounded_price)
            };
            let limit_order_unfilled_amount_before =
                next_initialized_tick.limit_order_unfilled_amount()?;
            if state.sqrt_price_next_x64 == swap_computed_result.sqrt_price_next_x64 {
                let limit_order_result = next_initialized_tick.match_limit_order(
                    state.amount_specified_remaining,
                    zero_for_one,
                    is_base_input,
                    total_fee_rate,
                    is_fee_on_input,
                )?;

                if limit_order_result.amount_in > 0 {
                    state.apply_swap_amounts(
                        limit_order_result.amount_in,
                        limit_order_result.amount_out,
                        limit_order_result.amm_fee_amount,
                        is_base_input,
                        is_fee_on_input,
                        amm_config.protocol_fee_rate,
                        amm_config.fund_fee_rate,
                    )?;
                }

                // Quote-path: skip `tick_array_current.update_initialized_tick_count(false)` and
                // `pool_state.flip_tick_array_bit(...)` (both mutate persistent state).

                if next_initialized_tick.has_liquidity()
                    && !next_initialized_tick.has_limit_orders()
                {
                    let fee_on_token0 = pool_state.is_fee_on_token0(zero_for_one);
                    // Note: we pass `pool_state.reward_infos` directly (no update_reward_infos)
                    // because the writes to `next_initialized_tick.reward_growths_outside_x64`
                    // are made on a local copy that's discarded — quote correctness depends
                    // only on `cross`'s returned liquidity_net.
                    let mut liquidity_net = next_initialized_tick.cross(
                        if fee_on_token0 {
                            state.fee_growth_global_x64
                        } else {
                            pool_state.fee_growth_global_0_x64
                        },
                        if fee_on_token0 {
                            pool_state.fee_growth_global_1_x64
                        } else {
                            state.fee_growth_global_x64
                        },
                        &pool_state.reward_infos,
                    );
                    if zero_for_one {
                        liquidity_net = liquidity_net.neg();
                    }
                    liquidity_next = liquidity_math::add_delta(state.liquidity, liquidity_net)?;
                }

                // Quote-path: skip `tick_array_current.update_tick_state(...)` (mutates the
                // tick array; not needed since `next_initialized_tick` is a discarded copy).

                state.tick = if (zero_for_one && !next_initialized_tick.has_limit_orders())
                    || (!zero_for_one && next_initialized_tick.has_limit_orders())
                {
                    state.tick_next - 1
                } else {
                    state.tick_next
                };
            } else if state.sqrt_price_x64 != swap_computed_result.sqrt_price_next_x64 {
                state.tick =
                    tick_math::get_tick_at_sqrt_price(swap_computed_result.sqrt_price_next_x64)?;
            }
            state.sqrt_price_x64 = swap_computed_result.sqrt_price_next_x64;
            if state.amount_specified_remaining == 0 || state.sqrt_price_x64 == target_price {
                let limit_order_unfilled_amount_after =
                    next_initialized_tick.limit_order_unfilled_amount()?;
                if state.amount_specified_remaining != 0
                    && limit_order_unfilled_amount_after != limit_order_unfilled_amount_before
                {
                    require_eq!(limit_order_unfilled_amount_after, 0);
                }
                break;
            }
            state.update_dynamic_fee_index(zero_for_one, is_skipped_tick_spacing)?;
        }
        state.liquidity = liquidity_next;
    }
    state.update_volatility_accumulator_on_price()?;

    // Quote-path: skip `observation_state.update(...)` and `pool_state.update_after_swap(...)`.

    let (amount_0, amount_1, _trade_fee_0, _trade_fee_1) = state.settle_amounts(
        amount_specified,
        zero_for_one,
        is_base_input,
        pool_state.is_fee_on_token0(zero_for_one),
    )?;

    Ok((state, amount_0, amount_1))
}

#[cfg(test)]
mod swap_test {
    use liquidity_math::get_delta_amounts_signed;
    use tick_array_bitmap_extension_test::{
        build_tick_array_bitmap_extension_info, BuildExtensionAccountInfo,
    };

    use super::*;
    use crate::states::pool_test::build_pool;
    use crate::states::tick_array_test::{
        build_tick, build_tick_array_with_tick_states, TickArrayInfo,
    };
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::vec;

    pub fn get_tick_array_states_mut(
        deque_tick_array_states: &VecDeque<RefCell<TickArrayState>>,
    ) -> RefCell<VecDeque<RefMut<TickArrayState>>> {
        let mut tick_array_states = VecDeque::new();

        for tick_array_state in deque_tick_array_states {
            tick_array_states.push_back(tick_array_state.borrow_mut());
        }
        RefCell::new(tick_array_states)
    }

    fn build_swap_param<'info>(
        tick_current: i32,
        tick_spacing: u16,
        sqrt_price_x64: u128,
        liquidity: u128,
        tick_array_infos: Vec<TickArrayInfo>,
    ) -> (
        AmmConfig,
        RefCell<PoolState>,
        VecDeque<RefCell<TickArrayState>>,
        RefCell<ObservationState>,
    ) {
        let amm_config = AmmConfig {
            trade_fee_rate: 1000,
            tick_spacing,
            ..Default::default()
        };
        let pool_state = build_pool(tick_current, tick_spacing, sqrt_price_x64, liquidity);

        let observation_state = RefCell::new(ObservationState::default());
        observation_state.borrow_mut().pool_id = pool_state.borrow().key();

        let mut tick_array_states: VecDeque<RefCell<TickArrayState>> = VecDeque::new();
        for tick_array_info in tick_array_infos {
            tick_array_states.push_back(build_tick_array_with_tick_states(
                pool_state.borrow().key(),
                tick_array_info.start_tick_index,
                tick_spacing,
                tick_array_info.ticks,
            ));
            pool_state
                .borrow_mut()
                .flip_tick_array_bit(None, tick_array_info.start_tick_index)
                .unwrap();
        }

        (amm_config, pool_state, tick_array_states, observation_state)
    }

    pub struct OpenPositionParam {
        pub amount_0: u64,
        pub amount_1: u64,
        // pub liquidity: u128,
        pub tick_lower: i32,
        pub tick_upper: i32,
    }

    fn setup_swap_test<'info>(
        start_tick: i32,
        tick_spacing: u16,
        position_params: Vec<OpenPositionParam>,
        zero_for_one: bool,
    ) -> (
        AmmConfig,
        RefCell<PoolState>,
        VecDeque<RefCell<TickArrayState>>,
        RefCell<ObservationState>,
        TickArrayBitmapExtension,
        u64,
        u64,
    ) {
        let amm_config = AmmConfig {
            trade_fee_rate: 1000,
            tick_spacing,
            ..Default::default()
        };

        let pool_state_refcel = build_pool(
            start_tick,
            tick_spacing,
            tick_math::get_sqrt_price_at_tick(start_tick).unwrap(),
            0,
        );

        let observation_state = RefCell::new(ObservationState::default());

        let param = &mut BuildExtensionAccountInfo::default();
        param.key = Pubkey::find_program_address(
            &[
                POOL_TICK_ARRAY_BITMAP_SEED.as_bytes(),
                pool_state_refcel.borrow().key().as_ref(),
            ],
            &crate::id(),
        )
        .0;
        let bitmap_extension = build_tick_array_bitmap_extension_info(param);
        let mut tick_array_states: VecDeque<RefCell<TickArrayState>> = VecDeque::new();
        let mut sum_amount_0: u64 = 0;
        let mut sum_amount_1: u64 = 0;
        {
            let mut pool_state = pool_state_refcel.borrow_mut();
            observation_state.borrow_mut().pool_id = pool_state.key();

            let mut tick_array_map = HashMap::new();

            for position_param in position_params {
                let liquidity = liquidity_math::get_liquidity_from_amounts(
                    pool_state.sqrt_price_x64,
                    tick_math::get_sqrt_price_at_tick(position_param.tick_lower).unwrap(),
                    tick_math::get_sqrt_price_at_tick(position_param.tick_upper).unwrap(),
                    position_param.amount_0,
                    position_param.amount_1,
                );

                let (amount_0, amount_1) = get_delta_amounts_signed(
                    start_tick,
                    tick_math::get_sqrt_price_at_tick(start_tick).unwrap(),
                    position_param.tick_lower,
                    position_param.tick_upper,
                    liquidity as i128,
                )
                .unwrap();
                sum_amount_0 += amount_0;
                sum_amount_1 += amount_1;
                let tick_array_lower_start_index =
                    TickArrayState::get_array_start_index(position_param.tick_lower, tick_spacing);

                if !tick_array_map.contains_key(&tick_array_lower_start_index) {
                    let mut tick_array_refcel = build_tick_array_with_tick_states(
                        pool_state.key(),
                        tick_array_lower_start_index,
                        tick_spacing,
                        vec![],
                    );
                    let tick_array_lower = tick_array_refcel.get_mut();

                    let tick_lower = tick_array_lower
                        .get_tick_state_mut(position_param.tick_lower, tick_spacing)
                        .unwrap();
                    tick_lower.tick = position_param.tick_lower;
                    tick_lower
                        .update(
                            pool_state.tick_current,
                            i128::try_from(liquidity).unwrap(),
                            0,
                            0,
                            false,
                            &[RewardInfo::default(); 3],
                        )
                        .unwrap();

                    tick_array_map.insert(tick_array_lower_start_index, tick_array_refcel);
                } else {
                    let tick_array_lower = tick_array_map
                        .get_mut(&tick_array_lower_start_index)
                        .unwrap();
                    let mut tick_array_lower_borrow_mut = tick_array_lower.borrow_mut();
                    let tick_lower = tick_array_lower_borrow_mut
                        .get_tick_state_mut(position_param.tick_lower, tick_spacing)
                        .unwrap();

                    tick_lower
                        .update(
                            pool_state.tick_current,
                            i128::try_from(liquidity).unwrap(),
                            0,
                            0,
                            false,
                            &[RewardInfo::default(); 3],
                        )
                        .unwrap();
                }
                let tick_array_upper_start_index =
                    TickArrayState::get_array_start_index(position_param.tick_upper, tick_spacing);
                if !tick_array_map.contains_key(&tick_array_upper_start_index) {
                    let mut tick_array_refcel = build_tick_array_with_tick_states(
                        pool_state.key(),
                        tick_array_upper_start_index,
                        tick_spacing,
                        vec![],
                    );
                    let tick_array_upper = tick_array_refcel.get_mut();

                    let tick_upper = tick_array_upper
                        .get_tick_state_mut(position_param.tick_upper, tick_spacing)
                        .unwrap();
                    tick_upper.tick = position_param.tick_upper;

                    tick_upper
                        .update(
                            pool_state.tick_current,
                            i128::try_from(liquidity).unwrap(),
                            0,
                            0,
                            true,
                            &[RewardInfo::default(); 3],
                        )
                        .unwrap();

                    tick_array_map.insert(tick_array_upper_start_index, tick_array_refcel);
                } else {
                    let tick_array_upper = tick_array_map
                        .get_mut(&tick_array_upper_start_index)
                        .unwrap();

                    let mut tick_array_upperr_borrow_mut = tick_array_upper.borrow_mut();
                    let tick_upper = tick_array_upperr_borrow_mut
                        .get_tick_state_mut(position_param.tick_upper, tick_spacing)
                        .unwrap();

                    tick_upper
                        .update(
                            pool_state.tick_current,
                            i128::try_from(liquidity).unwrap(),
                            0,
                            0,
                            true,
                            &[RewardInfo::default(); 3],
                        )
                        .unwrap();
                }
                if pool_state.tick_current >= position_param.tick_lower
                    && pool_state.tick_current < position_param.tick_upper
                {
                    pool_state.liquidity = liquidity_math::add_delta(
                        pool_state.liquidity,
                        i128::try_from(liquidity).unwrap(),
                    )
                    .unwrap();
                }
            }
            for (tickarray_start_index, tick_array_info) in tick_array_map {
                tick_array_states.push_back(tick_array_info);
                pool_state
                    .flip_tick_array_bit(Some(&bitmap_extension), tickarray_start_index)
                    .unwrap();
            }

            use std::convert::identity;
            if zero_for_one {
                tick_array_states.make_contiguous().sort_by(|a, b| {
                    identity(b.borrow().start_tick_index)
                        .cmp(&identity(a.borrow().start_tick_index))
                });
            } else {
                tick_array_states.make_contiguous().sort_by(|a, b| {
                    identity(a.borrow().start_tick_index)
                        .cmp(&identity(b.borrow().start_tick_index))
                });
            }
        }
        let bitmap_extension_state =
            *AccountLoader::<TickArrayBitmapExtension>::try_from(&bitmap_extension)
                .unwrap()
                .load()
                .unwrap()
                .deref();

        (
            amm_config,
            pool_state_refcel,
            tick_array_states,
            observation_state,
            bitmap_extension_state,
            sum_amount_0,
            sum_amount_1,
        )
    }

    #[cfg(test)]
    mod cross_tick_array_test {
        use super::*;

        #[test]
        fn zero_for_one_base_input_test() {
            let mut tick_current = -32395;
            let mut liquidity = 5124165121219;
            let mut sqrt_price_x64 = 3651942632306380802;
            let (amm_config, pool_state, mut tick_array_states, observation_state) =
                build_swap_param(
                    tick_current,
                    60,
                    sqrt_price_x64,
                    liquidity,
                    vec![
                        TickArrayInfo {
                            start_tick_index: -32400,
                            ticks: vec![
                                build_tick(-32400, 277065331032, -277065331032).take(),
                                build_tick(-29220, 1330680689, -1330680689).take(),
                                build_tick(-28860, 6408486554, -6408486554).take(),
                            ],
                        },
                        TickArrayInfo {
                            start_tick_index: -36000,
                            ticks: vec![
                                build_tick(-32460, 1194569667438, 536061033698).take(),
                                build_tick(-32520, 790917615645, 790917615645).take(),
                                build_tick(-32580, 152146472301, 128451145459).take(),
                                build_tick(-32640, 2625605835354, -1492054447712).take(),
                            ],
                        },
                    ],
                );

            // just cross the tickarray boundary(-32400), hasn't reached the next tick array initialized tick
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                12188240002,
                3049500711113990606,
                true,
                true,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current < tick_current);
            assert!(
                pool_state.borrow().tick_current > -32460
                    && pool_state.borrow().tick_current < -32400
            );
            assert!(pool_state.borrow().sqrt_price_x64 < sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity + 277065331032));
            assert!(amount_0 == 12188240002);

            tick_current = pool_state.borrow().tick_current;
            sqrt_price_x64 = pool_state.borrow().sqrt_price_x64;
            liquidity = pool_state.borrow().liquidity;

            // cross the tickarray boundary(-32400) in last step, now tickarray_current is the tickarray with start_index -36000,
            // so we pop the tickarray with start_index -32400
            // in this swap we will cross the tick(-32460), but not reach next tick (-32520)
            tick_array_states.pop_front();
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                121882400020,
                3049500711113990606,
                true,
                true,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current < tick_current);
            assert!(
                pool_state.borrow().tick_current > -32520
                    && pool_state.borrow().tick_current < -32460
            );
            assert!(pool_state.borrow().sqrt_price_x64 < sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity - 536061033698));
            assert!(amount_0 == 121882400020);

            tick_current = pool_state.borrow().tick_current;
            sqrt_price_x64 = pool_state.borrow().sqrt_price_x64;
            liquidity = pool_state.borrow().liquidity;

            // swap in tickarray with start_index -36000, cross the tick -32520
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                60941200010,
                3049500711113990606,
                true,
                true,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current < tick_current);
            assert!(
                pool_state.borrow().tick_current > -32580
                    && pool_state.borrow().tick_current < -32520
            );
            assert!(pool_state.borrow().sqrt_price_x64 < sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity - 790917615645));
            assert!(amount_0 == 60941200010);
        }

        #[test]
        fn zero_for_one_base_output_test() {
            let mut tick_current = -32395;
            let mut liquidity = 5124165121219;
            let mut sqrt_price_x64 = 3651942632306380802;
            let (amm_config, pool_state, mut tick_array_states, observation_state) =
                build_swap_param(
                    tick_current,
                    60,
                    sqrt_price_x64,
                    liquidity,
                    vec![
                        TickArrayInfo {
                            start_tick_index: -32400,
                            ticks: vec![
                                build_tick(-32400, 277065331032, -277065331032).take(),
                                build_tick(-29220, 1330680689, -1330680689).take(),
                                build_tick(-28860, 6408486554, -6408486554).take(),
                            ],
                        },
                        TickArrayInfo {
                            start_tick_index: -36000,
                            ticks: vec![
                                build_tick(-32460, 1194569667438, 536061033698).take(),
                                build_tick(-32520, 790917615645, 790917615645).take(),
                                build_tick(-32580, 152146472301, 128451145459).take(),
                                build_tick(-32640, 2625605835354, -1492054447712).take(),
                            ],
                        },
                    ],
                );

            // just cross the tickarray boundary(-32400), hasn't reached the next tick array initialized tick
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                477470480,
                3049500711113990606,
                true,
                false,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current < tick_current);
            assert!(
                pool_state.borrow().tick_current > -32460
                    && pool_state.borrow().tick_current < -32400
            );
            assert!(pool_state.borrow().sqrt_price_x64 < sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity + 277065331032));
            assert!(amount_1 == 477470480);

            tick_current = pool_state.borrow().tick_current;
            sqrt_price_x64 = pool_state.borrow().sqrt_price_x64;
            liquidity = pool_state.borrow().liquidity;

            // cross the tickarray boundary(-32400) in last step, now tickarray_current is the tickarray with start_index -36000,
            // so we pop the tickarray with start_index -32400
            // in this swap we will cross the tick(-32460), but not reach next tick (-32520)
            tick_array_states.pop_front();
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                4751002622,
                3049500711113990606,
                true,
                false,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current < tick_current);
            assert!(
                pool_state.borrow().tick_current > -32520
                    && pool_state.borrow().tick_current < -32460
            );
            assert!(pool_state.borrow().sqrt_price_x64 < sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity - 536061033698));
            assert!(amount_1 == 4751002622);

            tick_current = pool_state.borrow().tick_current;
            sqrt_price_x64 = pool_state.borrow().sqrt_price_x64;
            liquidity = pool_state.borrow().liquidity;

            // swap in tickarray with start_index -36000
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                2358130642,
                3049500711113990606,
                true,
                false,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current < tick_current);
            assert!(
                pool_state.borrow().tick_current > -32580
                    && pool_state.borrow().tick_current < -32520
            );
            assert!(pool_state.borrow().sqrt_price_x64 < sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity - 790917615645));
            assert!(amount_1 == 2358130642);
        }

        #[test]
        fn one_for_zero_base_input_test() {
            let mut tick_current = -32470;
            let mut liquidity = 5124165121219;
            let mut sqrt_price_x64 = 3638127228312488926;
            let (amm_config, pool_state, mut tick_array_states, observation_state) =
                build_swap_param(
                    tick_current,
                    60,
                    sqrt_price_x64,
                    liquidity,
                    vec![
                        TickArrayInfo {
                            start_tick_index: -36000,
                            ticks: vec![
                                build_tick(-32460, 1194569667438, 536061033698).take(),
                                build_tick(-32520, 790917615645, 790917615645).take(),
                                build_tick(-32580, 152146472301, 128451145459).take(),
                                build_tick(-32640, 2625605835354, -1492054447712).take(),
                            ],
                        },
                        TickArrayInfo {
                            start_tick_index: -32400,
                            ticks: vec![
                                build_tick(-32400, 277065331032, -277065331032).take(),
                                build_tick(-29220, 1330680689, -1330680689).take(),
                                build_tick(-28860, 6408486554, -6408486554).take(),
                            ],
                        },
                    ],
                );

            // just cross the tickarray boundary(-32460), hasn't reached the next tick array initialized tick
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                887470480,
                5882283448660210779,
                false,
                true,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current > tick_current);
            assert!(
                pool_state.borrow().tick_current > -32460
                    && pool_state.borrow().tick_current < -32400
            );
            assert!(pool_state.borrow().sqrt_price_x64 > sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity + 536061033698));
            assert!(amount_1 == 887470480);

            tick_current = pool_state.borrow().tick_current;
            sqrt_price_x64 = pool_state.borrow().sqrt_price_x64;
            liquidity = pool_state.borrow().liquidity;

            // cross the tickarray boundary(-32460) in last step, but not reached tick -32400, because -32400 is the next tickarray boundary,
            // so the tickarray_current still is the tick array with start_index -36000
            // in this swap we will cross the tick(-32400), but not reach next tick (-29220)
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                3087470480,
                5882283448660210779,
                false,
                true,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current > tick_current);
            assert!(
                pool_state.borrow().tick_current > -32400
                    && pool_state.borrow().tick_current < -29220
            );
            assert!(pool_state.borrow().sqrt_price_x64 > sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity - 277065331032));
            assert!(amount_1 == 3087470480);

            tick_current = pool_state.borrow().tick_current;
            sqrt_price_x64 = pool_state.borrow().sqrt_price_x64;
            liquidity = pool_state.borrow().liquidity;

            // swap in tickarray with start_index -32400, cross the tick -29220
            tick_array_states.pop_front();
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                200941200010,
                5882283448660210779,
                false,
                true,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current > tick_current);
            assert!(
                pool_state.borrow().tick_current > -29220
                    && pool_state.borrow().tick_current < -28860
            );
            assert!(pool_state.borrow().sqrt_price_x64 > sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity - 1330680689));
            assert!(amount_1 == 200941200010);
        }

        #[test]
        fn one_for_zero_base_output_test() {
            let mut tick_current = -32470;
            let mut liquidity = 5124165121219;
            let mut sqrt_price_x64 = 3638127228312488926;
            let (amm_config, pool_state, mut tick_array_states, observation_state) =
                build_swap_param(
                    tick_current,
                    60,
                    sqrt_price_x64,
                    liquidity,
                    vec![
                        TickArrayInfo {
                            start_tick_index: -36000,
                            ticks: vec![
                                build_tick(-32460, 1194569667438, 536061033698).take(),
                                build_tick(-32520, 790917615645, 790917615645).take(),
                                build_tick(-32580, 152146472301, 128451145459).take(),
                                build_tick(-32640, 2625605835354, -1492054447712).take(),
                            ],
                        },
                        TickArrayInfo {
                            start_tick_index: -32400,
                            ticks: vec![
                                build_tick(-32400, 277065331032, -277065331032).take(),
                                build_tick(-29220, 1330680689, -1330680689).take(),
                                build_tick(-28860, 6408486554, -6408486554).take(),
                            ],
                        },
                    ],
                );

            // just cross the tickarray boundary(-32460), hasn't reached the next tick array initialized tick
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                22796232052,
                5882283448660210779,
                false,
                false,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current > tick_current);
            assert!(
                pool_state.borrow().tick_current > -32460
                    && pool_state.borrow().tick_current < -32400
            );
            assert!(pool_state.borrow().sqrt_price_x64 > sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity + 536061033698));
            assert!(amount_0 == 22796232052);

            tick_current = pool_state.borrow().tick_current;
            sqrt_price_x64 = pool_state.borrow().sqrt_price_x64;
            liquidity = pool_state.borrow().liquidity;

            // cross the tickarray boundary(-32460) in last step, but not reached tick -32400, because -32400 is the next tickarray boundary,
            // so the tickarray_current still is the tick array with start_index -36000
            // in this swap we will cross the tick(-32400), but not reach next tick (-29220)
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                79023558189,
                5882283448660210779,
                false,
                false,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current > tick_current);
            assert!(
                pool_state.borrow().tick_current > -32400
                    && pool_state.borrow().tick_current < -29220
            );
            assert!(pool_state.borrow().sqrt_price_x64 > sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity - 277065331032));
            assert!(amount_0 == 79023558189);

            tick_current = pool_state.borrow().tick_current;
            sqrt_price_x64 = pool_state.borrow().sqrt_price_x64;
            liquidity = pool_state.borrow().liquidity;

            // swap in tickarray with start_index -32400, cross the tick -29220
            tick_array_states.pop_front();
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                4315086194758,
                5882283448660210779,
                false,
                false,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current > tick_current);
            assert!(
                pool_state.borrow().tick_current > -29220
                    && pool_state.borrow().tick_current < -28860
            );
            assert!(pool_state.borrow().sqrt_price_x64 > sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity - 1330680689));
            assert!(amount_0 == 4315086194758);
        }
    }

    #[cfg(test)]
    mod find_next_initialized_tick_test {
        use super::*;

        #[test]
        fn zero_for_one_current_tick_array_not_initialized_test() {
            let tick_current = -28776;
            let liquidity = 624165121219;
            let sqrt_price_x64 = tick_math::get_sqrt_price_at_tick(tick_current).unwrap();
            let (amm_config, pool_state, tick_array_states, observation_state) = build_swap_param(
                tick_current,
                60,
                sqrt_price_x64,
                liquidity,
                vec![TickArrayInfo {
                    start_tick_index: -32400,
                    ticks: vec![
                        build_tick(-32400, 277065331032, -277065331032).take(),
                        build_tick(-29220, 1330680689, -1330680689).take(),
                        build_tick(-28860, 6408486554, -6408486554).take(),
                    ],
                }],
            );

            // find the first initialzied tick(-28860) and cross it in tickarray
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                12188240002,
                tick_math::get_sqrt_price_at_tick(-32400).unwrap(),
                true,
                true,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current < tick_current);
            assert!(
                pool_state.borrow().tick_current > -29220
                    && pool_state.borrow().tick_current < -28860
            );
            assert!(pool_state.borrow().sqrt_price_x64 < sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity + 6408486554));
            assert!(amount_0 == 12188240002);
        }

        #[test]
        fn one_for_zero_current_tick_array_not_initialized_test() {
            let tick_current = -32405;
            let liquidity = 1224165121219;
            let sqrt_price_x64 = tick_math::get_sqrt_price_at_tick(tick_current).unwrap();
            let (amm_config, pool_state, tick_array_states, observation_state) = build_swap_param(
                tick_current,
                60,
                sqrt_price_x64,
                liquidity,
                vec![TickArrayInfo {
                    start_tick_index: -32400,
                    ticks: vec![
                        build_tick(-32400, 277065331032, -277065331032).take(),
                        build_tick(-29220, 1330680689, -1330680689).take(),
                        build_tick(-28860, 6408486554, -6408486554).take(),
                    ],
                }],
            );

            // find the first initialzied tick(-32400) and cross it in tickarray
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                12188240002,
                tick_math::get_sqrt_price_at_tick(-28860).unwrap(),
                false,
                true,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current > tick_current);
            assert!(
                pool_state.borrow().tick_current > -32400
                    && pool_state.borrow().tick_current < -29220
            );
            assert!(pool_state.borrow().sqrt_price_x64 > sqrt_price_x64);
            assert!(pool_state.borrow().liquidity == (liquidity - 277065331032));
            assert!(amount_1 == 12188240002);
        }
    }

    #[cfg(test)]
    mod liquidity_insufficient_test {
        use super::*;
        use crate::error::ErrorCode;
        #[test]
        fn no_enough_initialized_tickarray_in_pool_test() {
            let tick_current = -28776;
            let liquidity = 121219;
            let sqrt_price_x64 = tick_math::get_sqrt_price_at_tick(tick_current).unwrap();
            let (amm_config, pool_state, tick_array_states, observation_state) = build_swap_param(
                tick_current,
                60,
                sqrt_price_x64,
                liquidity,
                vec![TickArrayInfo {
                    start_tick_index: -32400,
                    ticks: vec![build_tick(-28860, 6408486554, -6408486554).take()],
                }],
            );

            let result = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                12188240002,
                tick_math::get_sqrt_price_at_tick(-32400).unwrap(),
                true,
                true,
                oracle::block_timestamp_mock() as u32,
            );
            assert!(result.is_err());
            assert_eq!(
                result.unwrap_err(),
                ErrorCode::MissingTickArrayBitmapExtensionAccount.into()
            );
        }
    }

    #[test]
    fn explain_why_zero_for_one_less_or_equal_current_tick() {
        let tick_current = -28859;
        let mut liquidity = 121219;
        let sqrt_price_x64 = tick_math::get_sqrt_price_at_tick(tick_current).unwrap();
        let (amm_config, pool_state, tick_array_states, observation_state) = build_swap_param(
            tick_current,
            60,
            sqrt_price_x64,
            liquidity,
            vec![TickArrayInfo {
                start_tick_index: -32400,
                ticks: vec![
                    build_tick(-32400, 277065331032, -277065331032).take(),
                    build_tick(-29220, 1330680689, -1330680689).take(),
                    build_tick(-28860, 6408486554, -6408486554).take(),
                ],
            }],
        );

        // not cross tick(-28860), but pool.tick_current = -28860
        let (amount_0, amount_1) = swap_internal(
            &amm_config,
            &mut pool_state.borrow_mut(),
            &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
            &mut observation_state.borrow_mut(),
            &None,
            25,
            tick_math::get_sqrt_price_at_tick(-32400).unwrap(),
            true,
            true,
            oracle::block_timestamp_mock() as u32,
        )
        .unwrap();
        println!("amount_0:{},amount_1:{}", amount_0, amount_1);
        assert!(pool_state.borrow().tick_current < tick_current);
        assert!(pool_state.borrow().tick_current == -28860);
        assert!(
            pool_state.borrow().sqrt_price_x64 > tick_math::get_sqrt_price_at_tick(-28860).unwrap()
        );
        assert!(pool_state.borrow().liquidity == liquidity);
        assert!(amount_0 == 25);

        // just cross tick(-28860), pool.tick_current = -28861
        let (amount_0, amount_1) = swap_internal(
            &amm_config,
            &mut pool_state.borrow_mut(),
            &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
            &mut observation_state.borrow_mut(),
            &None,
            3,
            tick_math::get_sqrt_price_at_tick(-32400).unwrap(),
            true,
            true,
            oracle::block_timestamp_mock() as u32,
        )
        .unwrap();
        println!("amount_0:{},amount_1:{}", amount_0, amount_1);
        assert!(pool_state.borrow().tick_current < tick_current);
        assert!(pool_state.borrow().tick_current == -28861);
        assert!(
            pool_state.borrow().sqrt_price_x64 > tick_math::get_sqrt_price_at_tick(-28861).unwrap()
        );
        assert!(pool_state.borrow().liquidity == liquidity + 6408486554);
        assert!(amount_0 == 3);

        liquidity = pool_state.borrow().liquidity;

        // we swap just a little amount, let pool tick_current also equal -28861
        // but pool.sqrt_price_x64 > tick_math::get_sqrt_price_at_tick(-28861)
        let (amount_0, amount_1) = swap_internal(
            &amm_config,
            &mut pool_state.borrow_mut(),
            &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
            &mut observation_state.borrow_mut(),
            &None,
            50,
            tick_math::get_sqrt_price_at_tick(-32400).unwrap(),
            true,
            true,
            oracle::block_timestamp_mock() as u32,
        )
        .unwrap();
        println!("amount_0:{},amount_1:{}", amount_0, amount_1);
        assert!(pool_state.borrow().tick_current == -28861);
        assert!(
            pool_state.borrow().sqrt_price_x64 > tick_math::get_sqrt_price_at_tick(-28861).unwrap()
        );
        assert!(pool_state.borrow().liquidity == liquidity);
        assert!(amount_0 == 50);
    }

    #[cfg(test)]
    mod swap_edge_test {
        use super::*;

        #[test]
        fn zero_for_one_swap_edge_case() {
            let mut tick_current = -28859;
            let liquidity = 121219;
            let mut sqrt_price_x64 = tick_math::get_sqrt_price_at_tick(tick_current).unwrap();
            let (amm_config, pool_state, tick_array_states, observation_state) = build_swap_param(
                tick_current,
                60,
                sqrt_price_x64,
                liquidity,
                vec![
                    TickArrayInfo {
                        start_tick_index: -32400,
                        ticks: vec![
                            build_tick(-32400, 277065331032, -277065331032).take(),
                            build_tick(-29220, 1330680689, -1330680689).take(),
                            build_tick(-28860, 6408486554, -6408486554).take(),
                        ],
                    },
                    TickArrayInfo {
                        start_tick_index: -28800,
                        ticks: vec![build_tick(-28800, 3726362727, -3726362727).take()],
                    },
                ],
            );

            // zero for one, just cross tick(-28860),  pool.tick_current = -28861 and pool.sqrt_price_x64 = tick_math::get_sqrt_price_at_tick(-28860)
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                27,
                tick_math::get_sqrt_price_at_tick(-32400).unwrap(),
                true,
                true,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current < tick_current);
            assert!(pool_state.borrow().tick_current == -28861);
            assert!(
                pool_state.borrow().sqrt_price_x64
                    == tick_math::get_sqrt_price_at_tick(-28860).unwrap()
            );
            assert!(pool_state.borrow().liquidity == liquidity + 6408486554);
            assert!(amount_0 == 27);

            tick_current = pool_state.borrow().tick_current;
            sqrt_price_x64 = pool_state.borrow().sqrt_price_x64;

            // we swap just a little amount, it is completely taken by fees, the sqrt price and the tick will remain the same
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                1,
                tick_math::get_sqrt_price_at_tick(-32400).unwrap(),
                true,
                true,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current == tick_current);
            assert!(pool_state.borrow().tick_current == -28861);
            assert!(pool_state.borrow().sqrt_price_x64 == sqrt_price_x64);

            tick_current = pool_state.borrow().tick_current;
            sqrt_price_x64 = pool_state.borrow().sqrt_price_x64;

            // reverse swap direction, one_for_zero
            // Actually, the loop for this swap was executed twice because the previous swap happened to have `pool.tick_current` exactly on the boundary that is divisible by `tick_spacing`.
            // In the first iteration of this swap's loop, it found the initial tick (-28860), but at this point, both the initial and final prices were equal to the price at tick -28860.
            // This did not meet the conditions for swapping so both swap_amount_input and swap_amount_output were 0. The actual output was calculated in the second iteration of the loop.
            let (amount_0, amount_1) = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &None,
                10,
                tick_math::get_sqrt_price_at_tick(-28800).unwrap(),
                false,
                true,
                oracle::block_timestamp_mock() as u32,
            )
            .unwrap();
            println!("amount_0:{},amount_1:{}", amount_0, amount_1);
            assert!(pool_state.borrow().tick_current > tick_current);
            assert!(pool_state.borrow().sqrt_price_x64 > sqrt_price_x64);
            assert!(
                pool_state.borrow().tick_current > -28860
                    && pool_state.borrow().tick_current <= -28800
            );
        }
    }

    #[cfg(test)]
    mod sqrt_price_limit_optimization_min_specified_test {
        use super::*;
        #[test]
        fn zero_for_one_base_input_with_min_amount_specified() {
            let tick_spacing = 10;
            let zero_for_one = true;
            let is_base_input = true;
            let tick_lower = tick_math::MIN_TICK + 1;
            let tick_upper = tick_math::MAX_TICK - 1;
            let tick_current = 0;
            let amount_0 = u64::MAX - 1;
            let amount_1 = u64::MAX - 1;

            let (
                amm_config,
                pool_state,
                tick_array_states,
                observation_state,
                bitmap_extension_state,
                sum_amount_0,
                sum_amount_1,
            ) = setup_swap_test(
                tick_current,
                tick_spacing as u16,
                vec![OpenPositionParam {
                    amount_0: amount_0,
                    amount_1: amount_1,
                    tick_lower: tick_lower,
                    tick_upper: tick_upper,
                }],
                zero_for_one,
            );
            println!(
                "sum_amount_0: {}, sum_amount_1: {}",
                sum_amount_0, sum_amount_1,
            );
            let amount_specified = 1;
            let result = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &Some(bitmap_extension_state),
                amount_specified,
                tick_math::MIN_SQRT_PRICE_X64 + 1,
                zero_for_one,
                is_base_input,
                1,
            );
            println!("{:#?}", result);
            let pool = pool_state.borrow();
            let sqrt_price_x64 = pool.sqrt_price_x64;
            let sqrt_price = sqrt_price_x64 as f64 / fixed_point_64::Q64 as f64;
            println!("price: {}", sqrt_price * sqrt_price);
        }

        #[test]
        fn zero_for_one_base_out_with_min_amount_specified() {
            let tick_spacing = 10;
            let zero_for_one = true;
            let is_base_input = false;
            let tick_lower = tick_math::MIN_TICK + 1;
            let tick_upper = tick_math::MAX_TICK - 1;
            let tick_current = 0;
            let amount_0 = u64::MAX - 1;
            let amount_1 = u64::MAX - 1;

            let (
                amm_config,
                pool_state,
                tick_array_states,
                observation_state,
                bitmap_extension_state,
                sum_amount_0,
                sum_amount_1,
            ) = setup_swap_test(
                tick_current,
                tick_spacing as u16,
                vec![OpenPositionParam {
                    amount_0: amount_0,
                    amount_1: amount_1,
                    tick_lower: tick_lower,
                    tick_upper: tick_upper,
                }],
                zero_for_one,
            );
            println!(
                "sum_amount_0: {}, sum_amount_1: {}",
                sum_amount_0, sum_amount_1,
            );
            let amount_specified = 1;
            let result = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &Some(bitmap_extension_state),
                amount_specified,
                tick_math::MIN_SQRT_PRICE_X64 + 1,
                zero_for_one,
                is_base_input,
                1,
            );
            println!("{:#?}", result);
            let pool = pool_state.borrow();
            let sqrt_price_x64 = pool.sqrt_price_x64;
            let sqrt_price = sqrt_price_x64 as f64 / fixed_point_64::Q64 as f64;
            println!("price: {}", sqrt_price * sqrt_price);
        }

        #[test]
        fn one_for_zero_base_in_with_min_amount_specified() {
            let tick_spacing = 10;
            let zero_for_one = false;
            let is_base_input = true;
            let tick_lower = tick_math::MIN_TICK + 1;
            let tick_upper = tick_math::MAX_TICK - 1;
            let tick_current = 0;
            let amount_0 = u64::MAX - 1;
            let amount_1 = u64::MAX - 1;

            let (
                amm_config,
                pool_state,
                tick_array_states,
                observation_state,
                bitmap_extension_state,
                sum_amount_0,
                sum_amount_1,
            ) = setup_swap_test(
                tick_current,
                tick_spacing as u16,
                vec![OpenPositionParam {
                    amount_0: amount_0,
                    amount_1: amount_1,
                    tick_lower: tick_lower,
                    tick_upper: tick_upper,
                }],
                zero_for_one,
            );
            println!(
                "sum_amount_0: {}, sum_amount_1: {}",
                sum_amount_0, sum_amount_1,
            );
            let amount_specified = 1;
            let result = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &Some(bitmap_extension_state),
                amount_specified,
                tick_math::MAX_SQRT_PRICE_X64 - 1,
                zero_for_one,
                is_base_input,
                1,
            );
            println!("{:#?}", result);
            let pool = pool_state.borrow();
            let sqrt_price_x64 = pool.sqrt_price_x64;
            let sqrt_price = sqrt_price_x64 as f64 / fixed_point_64::Q64 as f64;
            println!("price: {}", sqrt_price * sqrt_price);
        }
        #[test]
        fn one_for_zero_base_out_with_min_amount_specified() {
            let tick_spacing = 10;
            let zero_for_one = false;
            let is_base_input = false;
            let tick_lower = tick_math::MIN_TICK + 1;
            let tick_upper = tick_math::MAX_TICK - 1;
            let tick_current = 0;
            let amount_0 = u64::MAX - 1;
            let amount_1 = u64::MAX - 1;

            let (
                amm_config,
                pool_state,
                tick_array_states,
                observation_state,
                bitmap_extension_state,
                sum_amount_0,
                sum_amount_1,
            ) = setup_swap_test(
                tick_current,
                tick_spacing as u16,
                vec![OpenPositionParam {
                    amount_0: amount_0,
                    amount_1: amount_1,
                    tick_lower: tick_lower,
                    tick_upper: tick_upper,
                }],
                zero_for_one,
            );
            println!(
                "sum_amount_0: {}, sum_amount_1: {}",
                sum_amount_0, sum_amount_1,
            );
            let amount_specified = 1;
            let result = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &Some(bitmap_extension_state),
                amount_specified,
                tick_math::MAX_SQRT_PRICE_X64 - 1,
                zero_for_one,
                is_base_input,
                1,
            );
            println!("{:#?}", result);
            let pool = pool_state.borrow();
            let sqrt_price_x64 = pool.sqrt_price_x64;
            let sqrt_price = sqrt_price_x64 as f64 / fixed_point_64::Q64 as f64;
            println!("price: {}", sqrt_price * sqrt_price);
        }
    }
    #[cfg(test)]
    mod sqrt_price_limit_optimization_max_specified_test {
        use super::*;
        #[test]
        fn zero_for_one_base_input_with_max_amount_specified() {
            let tick_spacing = 10;
            let zero_for_one = true;
            let is_base_input = true;
            let tick_lower = tick_math::MIN_TICK + 1;
            let tick_upper = tick_math::MAX_TICK - 1;
            let tick_current = 0;
            let amount_0 = u64::MAX / 2;
            let amount_1 = u64::MAX / 2;

            let (
                amm_config,
                pool_state,
                tick_array_states,
                observation_state,
                bitmap_extension_state,
                sum_amount_0,
                sum_amount_1,
            ) = setup_swap_test(
                tick_current,
                tick_spacing as u16,
                vec![OpenPositionParam {
                    amount_0: amount_0,
                    amount_1: amount_1,
                    tick_lower: tick_lower,
                    tick_upper: tick_upper,
                }],
                zero_for_one,
            );
            println!(
                "sum_amount_0: {}, sum_amount_1: {}",
                sum_amount_0, sum_amount_1,
            );
            let amount_specified = u64::MAX / 2;
            let result = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &Some(bitmap_extension_state),
                amount_specified,
                tick_math::MIN_SQRT_PRICE_X64 + 1,
                zero_for_one,
                is_base_input,
                1,
            );
            println!("{:#?}", result);
            let pool = pool_state.borrow();
            let sqrt_price_x64 = pool.sqrt_price_x64;
            let sqrt_price = sqrt_price_x64 as f64 / fixed_point_64::Q64 as f64;
            println!("price: {}", sqrt_price * sqrt_price);
        }

        #[test]
        fn zero_for_one_base_out_with_max_amount_specified() {
            let tick_spacing = 10;
            let zero_for_one = true;
            let is_base_input = false;
            let tick_lower = tick_math::MIN_TICK + 1;
            let tick_upper = tick_math::MAX_TICK - 1;
            let tick_current = 0;
            let amount_0 = u64::MAX / 2;
            let amount_1 = u64::MAX / 2;

            let (
                amm_config,
                pool_state,
                tick_array_states,
                observation_state,
                bitmap_extension_state,
                sum_amount_0,
                sum_amount_1,
            ) = setup_swap_test(
                tick_current,
                tick_spacing as u16,
                vec![OpenPositionParam {
                    amount_0: amount_0,
                    amount_1: amount_1,
                    tick_lower: tick_lower,
                    tick_upper: tick_upper,
                }],
                zero_for_one,
            );
            println!(
                "sum_amount_0: {}, sum_amount_1: {}",
                sum_amount_0, sum_amount_1,
            );
            let amount_specified = u64::MAX / 4;
            let result = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &Some(bitmap_extension_state),
                amount_specified,
                tick_math::MIN_SQRT_PRICE_X64 + 1,
                zero_for_one,
                is_base_input,
                1,
            );
            println!("{:#?}", result);
            let pool = pool_state.borrow();
            let sqrt_price_x64 = pool.sqrt_price_x64;
            let sqrt_price = sqrt_price_x64 as f64 / fixed_point_64::Q64 as f64;
            println!("price: {}", sqrt_price * sqrt_price);
        }

        #[test]
        fn one_for_zero_base_in_with_max_amount_specified() {
            let tick_spacing = 10;
            let zero_for_one = false;
            let is_base_input = true;
            let tick_lower = tick_math::MIN_TICK + 1;
            let tick_upper = tick_math::MAX_TICK - 1;
            let tick_current = 0;
            let amount_0 = u64::MAX / 2;
            let amount_1 = u64::MAX / 2;

            let (
                amm_config,
                pool_state,
                tick_array_states,
                observation_state,
                bitmap_extension_state,
                sum_amount_0,
                sum_amount_1,
            ) = setup_swap_test(
                tick_current,
                tick_spacing as u16,
                vec![OpenPositionParam {
                    amount_0: amount_0,
                    amount_1: amount_1,
                    tick_lower: tick_lower,
                    tick_upper: tick_upper,
                }],
                zero_for_one,
            );
            println!(
                "sum_amount_0: {}, sum_amount_1: {}",
                sum_amount_0, sum_amount_1,
            );
            let amount_specified = u64::MAX / 2;
            let result = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &Some(bitmap_extension_state),
                amount_specified,
                tick_math::MAX_SQRT_PRICE_X64 - 1,
                zero_for_one,
                is_base_input,
                1,
            );
            println!("{:#?}", result);
            let pool = pool_state.borrow();
            let sqrt_price_x64 = pool.sqrt_price_x64;
            let sqrt_price = sqrt_price_x64 as f64 / fixed_point_64::Q64 as f64;
            println!("price: {}", sqrt_price * sqrt_price);
        }
        #[test]
        fn one_for_zero_base_out_with_min_amount_specified() {
            let tick_spacing = 10;
            let zero_for_one = false;
            let is_base_input = false;
            let tick_lower = tick_math::MIN_TICK + 1;
            let tick_upper = tick_math::MAX_TICK - 1;
            let tick_current = 0;
            let amount_0 = u64::MAX / 2;
            let amount_1 = u64::MAX / 2;

            let (
                amm_config,
                pool_state,
                tick_array_states,
                observation_state,
                bitmap_extension_state,
                sum_amount_0,
                sum_amount_1,
            ) = setup_swap_test(
                tick_current,
                tick_spacing as u16,
                vec![OpenPositionParam {
                    amount_0: amount_0,
                    amount_1: amount_1,
                    tick_lower: tick_lower,
                    tick_upper: tick_upper,
                }],
                zero_for_one,
            );
            println!(
                "sum_amount_0: {}, sum_amount_1: {}",
                sum_amount_0, sum_amount_1,
            );
            let amount_specified = u64::MAX / 4;
            let result = swap_internal(
                &amm_config,
                &mut pool_state.borrow_mut(),
                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                &mut observation_state.borrow_mut(),
                &Some(bitmap_extension_state),
                amount_specified,
                tick_math::MAX_SQRT_PRICE_X64 - 1,
                zero_for_one,
                is_base_input,
                1,
            );
            println!("{:#?}", result);
            let pool = pool_state.borrow();
            let sqrt_price_x64 = pool.sqrt_price_x64;
            let sqrt_price = sqrt_price_x64 as f64 / fixed_point_64::Q64 as f64;
            println!("price: {}", sqrt_price * sqrt_price);
        }
    }
    #[cfg(test)]
    mod sqrt_price_limit_optimization_test {
        use super::*;
        use proptest::prelude::*;
        use std::{convert::identity, u64};

        use proptest::prop_assume;
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(2048))]

            #[test]
            fn zero_for_one_base_input_test(
                tick_current in tick_math::MIN_TICK..tick_math::MAX_TICK,
                amount_0 in 1000000..u64::MAX,
                amount_1 in 1000000..u64::MAX,
                tick_lower in (tick_math::MIN_TICK..=tick_math::MAX_TICK).prop_filter("Must be multiple of 10", |x| x % 10 == 0),
                tick_upper in (tick_math::MIN_TICK..=tick_math::MAX_TICK).prop_filter("Must be multiple of 10", |x| x % 10 == 0),
            ){
                let tick_spacing = 10;
                let zero_for_one = true;
                let is_base_input = true;
                if tick_lower%tick_spacing ==0 && tick_upper%tick_spacing ==0 && tick_upper>tick_lower{

                    let (amm_config, pool_state, tick_array_states, observation_state,bitmap_extension_state,  sum_amount_0, sum_amount_1) = setup_swap_test(
                        tick_current,
                        tick_spacing as u16,
                        vec![OpenPositionParam{amount_0:amount_0,amount_1:amount_1, tick_lower:tick_lower, tick_upper:tick_upper}],
                        zero_for_one
                        );

                    prop_assume!(sum_amount_1 > 1);
                    let mut rng = rand::thread_rng();
                    let amount_specified  = rng.gen_range(1..u64::MAX - sum_amount_0);

                    let result = swap_internal(
                        &amm_config,
                        &mut pool_state.borrow_mut(),
                        &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                        &mut observation_state.borrow_mut(),
                        &Some(bitmap_extension_state),
                        amount_specified,
                        tick_math::MIN_SQRT_PRICE_X64 + 1,
                        zero_for_one,
                        is_base_input,
                        0,
                    );

                    if result.is_ok() {
                        let ( amount_0_before, amount_1_before) = result.unwrap();

                        let (amm_config, pool_state, tick_array_states, observation_state,bitmap_extension_state,  _sum_amount_0, _sum_amount_1) = setup_swap_test(
                            tick_current,
                            tick_spacing as u16,
                            vec![OpenPositionParam{amount_0:amount_0,amount_1:amount_1, tick_lower:tick_lower, tick_upper:tick_upper}],
                            zero_for_one
                        );
                        let result = swap_internal(
                            &amm_config,
                            &mut pool_state.borrow_mut(),
                            &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                            &mut observation_state.borrow_mut(),
                            &Some(bitmap_extension_state),
                            amount_specified,
                            tick_math::MIN_SQRT_PRICE_X64 + 1,
                            zero_for_one,
                            is_base_input,
                            oracle::block_timestamp_mock() as u32,
                        );
                        assert!(result.is_ok());

                        // println!("----- input: tick_current:{}, amount_0:{}, amount_1:{}, amount_specified:{},tick_lower:{}, tick_upper:{},liquidity:{}", tick_current, amount_0, amount_1,amount_specified, tick_lower, tick_upper, identity(pool_state.borrow().liquidity));

                        let ( amount_0_after, amount_1_after) = result.unwrap();
                        assert_eq!(amount_0_before, amount_0_after);
                        assert_eq!(amount_1_before, amount_1_after);

                    }else{
                        let err =  result.err().unwrap();
                        if err == crate::error::ErrorCode::MaxTokenOverflow.into(){
                            println!("##### original swap is overflow ");
                            let result = swap_internal(
                                &amm_config,
                                &mut pool_state.borrow_mut(),
                                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                                &mut observation_state.borrow_mut(),
                                &Some(bitmap_extension_state),
                                amount_specified,
                                tick_math::MIN_SQRT_PRICE_X64 + 1,
                                zero_for_one,
                                is_base_input,
                                oracle::block_timestamp_mock() as u32,
                            );
                            if result.is_err(){
                                println!("{:#?}", result);
                            }
                        }else{
                            println!("{}", err);
                        }
                    }
                }
            }

            #[test]
            fn zero_for_one_base_output_test(
                tick_current in tick_math::MIN_TICK..tick_math::MAX_TICK,
                amount_0 in 1000000..u64::MAX,
                amount_1 in 1000000..u64::MAX,
                tick_lower in (tick_math::MIN_TICK..=tick_math::MAX_TICK).prop_filter("Must be multiple of 100", |x| x % 10 == 0),
                tick_upper in (tick_math::MIN_TICK..=tick_math::MAX_TICK).prop_filter("Must be multiple of 100", |x| x % 10 == 0),
            ){
                let tick_spacing = 10;
                let zero_for_one = true;
                let base_input= false;
                if tick_lower%tick_spacing ==0 && tick_upper%tick_spacing ==0 && tick_upper>tick_lower{
                    let (amm_config, pool_state, tick_array_states, observation_state,bitmap_extension_state, _sum_amount_0, sum_amount_1) = setup_swap_test(
                        tick_current,
                        tick_spacing as u16,
                        vec![OpenPositionParam{amount_0:amount_0,amount_1:amount_1, tick_lower:tick_lower, tick_upper:tick_upper}],
                        zero_for_one
                    );

                    prop_assume!(sum_amount_1 > 1);
                    let mut rng = rand::thread_rng();
                    let amount_specified  = rng.gen_range(1..sum_amount_1);
                    // println!("----- input: tick_current:{}, amount_0:{}, amount_1:{}, amount_specified:{},tick_lower:{}, tick_upper:{}", tick_current, amount_0, amount_1,amount_specified, tick_lower, tick_upper);
                    let result = swap_internal(
                        &amm_config,
                        &mut pool_state.borrow_mut(),
                        &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                        &mut observation_state.borrow_mut(),
                        &Some(bitmap_extension_state),
                        amount_specified,
                        tick_math::MIN_SQRT_PRICE_X64 + 1,
                        zero_for_one,
                        base_input,
                        0,
                    );

                    if result.is_ok() {
                        let ( amount_0_before, amount_1_before) = result.unwrap();

                        let (amm_config, pool_state, tick_array_states, observation_state,bitmap_extension_state, _sum_amount_0, _sum_amount_1) = setup_swap_test(
                            tick_current,
                            tick_spacing as u16,
                            vec![OpenPositionParam{amount_0:amount_0,amount_1:amount_1, tick_lower:tick_lower, tick_upper:tick_upper}],
                            zero_for_one
                        );
                        let result = swap_internal(
                            &amm_config,
                            &mut pool_state.borrow_mut(),
                            &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                            &mut observation_state.borrow_mut(),
                            &Some(bitmap_extension_state),
                            amount_specified,
                            tick_math::MIN_SQRT_PRICE_X64 + 1,
                            zero_for_one,
                            base_input,
                            oracle::block_timestamp_mock() as u32,
                        );
                        assert!(result.is_ok());

                        println!("----- input: tick_current:{}, amount_0:{}, amount_1:{}, amount_specified:{},tick_lower:{}, tick_upper:{},liquidity:{}", tick_current, amount_0, amount_1,amount_specified, tick_lower, tick_upper, identity(pool_state.borrow().liquidity));

                        let ( amount_0_after, amount_1_after) = result.unwrap();
                        assert_eq!(amount_0_before, amount_0_after);
                        assert_eq!(amount_1_before, amount_1_after);

                    }else{
                        let err =  result.err().unwrap();
                        if err == crate::error::ErrorCode::MaxTokenOverflow.into(){
                            println!("##### original swap is overflow");
                            let result = swap_internal(
                                &amm_config,
                                &mut pool_state.borrow_mut(),
                                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                                &mut observation_state.borrow_mut(),
                                &Some(bitmap_extension_state),
                                amount_specified,
                                tick_math::MIN_SQRT_PRICE_X64 + 1,
                                zero_for_one,
                                base_input,
                                oracle::block_timestamp_mock() as u32,
                            );
                            if result.is_err(){
                                println!("{:#?}", result);
                            }
                        }else{
                            println!("{}", err);
                        }
                    }
                }
            }

            #[test]
            fn one_for_zero_base_input_test(
                tick_current in tick_math::MIN_TICK..tick_math::MAX_TICK,
                amount_0 in 1000000..u64::MAX,
                amount_1 in 1000000..u64::MAX,
                tick_lower in (tick_math::MIN_TICK..=tick_math::MAX_TICK).prop_filter("Must be multiple of 100", |x| x % 10 == 0),
                tick_upper in (tick_math::MIN_TICK..=tick_math::MAX_TICK).prop_filter("Must be multiple of 100", |x| x % 10 == 0),
            ){
                let tick_spacing = 10;
                let zero_for_one = false;
                let is_base_input = true;
                if tick_lower%tick_spacing ==0 && tick_upper%tick_spacing ==0 && tick_current>tick_lower && tick_current<tick_upper{
                    // println!("----- input: tick_current:{}, amount_0:{}, amount_1:{}, amount_specified:{},tick_lower:{}, tick_upper:{}", tick_current, amount_0, amount_1,amount_specified, tick_lower, tick_upper);
                    let (amm_config, pool_state, tick_array_states, observation_state,bitmap_extension_state,  sum_amount_0, sum_amount_1) = setup_swap_test(
                        tick_current,
                        tick_spacing as u16,
                        vec![OpenPositionParam{amount_0:amount_0,amount_1:amount_1, tick_lower:tick_lower, tick_upper:tick_upper}],
                        zero_for_one
                    );

                    prop_assume!(sum_amount_0 > 1);
                    let mut rng = rand::thread_rng();
                    let amount_specified  = rng.gen_range(1..u64::MAX - sum_amount_1);

                    let result = swap_internal(
                        &amm_config,
                        &mut pool_state.borrow_mut(),
                        &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                        &mut observation_state.borrow_mut(),
                        &Some(bitmap_extension_state),
                        amount_specified,
                        tick_math::MAX_SQRT_PRICE_X64 - 1,
                        zero_for_one,
                        is_base_input,
                        0,
                    );


                    if result.is_ok() {
                        let ( amount_0_before, amount_1_before) = result.unwrap();

                        let (amm_config, pool_state, tick_array_states, observation_state,bitmap_extension_state,  _sum_amount_0, _sum_amount_1) = setup_swap_test(
                            tick_current,
                            tick_spacing as u16,
                            vec![OpenPositionParam{amount_0:amount_0,amount_1:amount_1, tick_lower:tick_lower, tick_upper:tick_upper}],
                            zero_for_one
                        );
                        let result = swap_internal(
                            &amm_config,
                            &mut pool_state.borrow_mut(),
                            &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                            &mut observation_state.borrow_mut(),
                            &Some(bitmap_extension_state),
                            amount_specified,
                            tick_math::MAX_SQRT_PRICE_X64 - 1,
                            zero_for_one,
                            is_base_input,
                            oracle::block_timestamp_mock() as u32,
                        );
                        assert!(result.is_ok());

                        // println!("----- input: tick_current:{}, amount_0:{}, amount_1:{}, amount_specified:{},tick_lower:{}, tick_upper:{},liquidity:{}", tick_current, amount_0, amount_1,amount_specified, tick_lower, tick_upper, identity(pool_state.borrow().liquidity));

                        let (amount_0_after, amount_1_after) = result.unwrap();
                        assert_eq!(amount_0_before, amount_0_after);
                        assert_eq!(amount_1_before, amount_1_after);

                    }else {
                        let err =  result.err().unwrap();
                        if err == crate::error::ErrorCode::MaxTokenOverflow.into(){
                            // println!("##### original swap is overflow ");
                            let _result = swap_internal(
                                &amm_config,
                                &mut pool_state.borrow_mut(),
                                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                                &mut observation_state.borrow_mut(),
                                &Some(bitmap_extension_state),
                                amount_specified,
                                tick_math::MAX_SQRT_PRICE_X64 - 1,
                                zero_for_one,
                                is_base_input,
                                oracle::block_timestamp_mock() as u32,
                            );

                        }else{
                            println!("{}", err);
                        }
                    }
                }
            }

            #[test]
            fn one_for_zero_base_output_test(
                tick_current in tick_math::MIN_TICK..tick_math::MAX_TICK,
                amount_0 in 1000000..u64::MAX,
                amount_1 in 1000000..u64::MAX,
                tick_lower in (tick_math::MIN_TICK..=tick_math::MAX_TICK).prop_filter("Must be multiple of 100", |x| x % 10 == 0),
                tick_upper in (tick_math::MIN_TICK..=tick_math::MAX_TICK).prop_filter("Must be multiple of 100", |x| x % 10 == 0),
            ){
                let tick_spacing = 10;
                let zero_for_one = false;
                let is_base_input = false;
                if tick_lower%tick_spacing ==0 && tick_upper%tick_spacing ==0 && tick_current>tick_lower && tick_current<tick_upper{

                    // println!("----- input: tick_current:{}, amount_0:{}, amount_1:{}, amount_specified:{},tick_lower:{}, tick_upper:{}", tick_current, amount_0, amount_1,amount_specified, tick_lower, tick_upper);
                    let (amm_config, pool_state, tick_array_states, observation_state,bitmap_extension_state,  sum_amount_0, _sum_amount_1) = setup_swap_test(
                        tick_current,
                        tick_spacing as u16,
                        vec![OpenPositionParam{amount_0:amount_0,amount_1:amount_1, tick_lower:tick_lower, tick_upper:tick_upper}],
                        zero_for_one
                    );
                    prop_assume!(sum_amount_0 > 1);
                    let mut rng = rand::thread_rng();
                    let amount_specified  = rng.gen_range(1..sum_amount_0);

                    let result = swap_internal(
                        &amm_config,
                        &mut pool_state.borrow_mut(),
                        &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                        &mut observation_state.borrow_mut(),
                        &Some(bitmap_extension_state),
                        amount_specified,
                        tick_math::MAX_SQRT_PRICE_X64 - 1,
                        zero_for_one,
                        is_base_input,
                        0,
                    );

                    if result.is_ok() {
                        let ( amount_0_before, amount_1_before) = result.unwrap();

                        let (amm_config, pool_state, tick_array_states, observation_state,bitmap_extension_state,  _sum_amount_0, _sum_amount_1) = setup_swap_test(
                            tick_current,
                            tick_spacing as u16,
                            vec![OpenPositionParam{amount_0:amount_0,amount_1:amount_1, tick_lower:tick_lower, tick_upper:tick_upper}],
                            zero_for_one
                        );
                        let result = swap_internal(
                            &amm_config,
                            &mut pool_state.borrow_mut(),
                            &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                            &mut observation_state.borrow_mut(),
                            &Some(bitmap_extension_state),
                            amount_specified,
                            tick_math::MAX_SQRT_PRICE_X64 - 1,
                            zero_for_one,
                            is_base_input,
                            oracle::block_timestamp_mock() as u32,
                        );
                        assert!(result.is_ok());

                        // println!("----- input: tick_current:{}, amount_0:{}, amount_1:{}, amount_specified:{},tick_lower:{}, tick_upper:{},liquidity:{}", tick_current, amount_0, amount_1,amount_specified, tick_lower, tick_upper, identity(pool_state.borrow().liquidity));

                        let (amount_0_after, amount_1_after) = result.unwrap();
                        assert_eq!(amount_0_before, amount_0_after);
                        assert_eq!(amount_1_before, amount_1_after);

                    }else {
                        let err =  result.err().unwrap();
                        if err == crate::error::ErrorCode::MaxTokenOverflow.into(){
                            println!("##### original swap is overflow ");
                            let _result = swap_internal(
                                &amm_config,
                                &mut pool_state.borrow_mut(),
                                &mut get_tick_array_states_mut(&tick_array_states).borrow_mut(),
                                &mut observation_state.borrow_mut(),
                                &Some(bitmap_extension_state),
                                amount_specified,
                                tick_math::MAX_SQRT_PRICE_X64 - 1,
                                zero_for_one,
                                is_base_input,
                                oracle::block_timestamp_mock() as u32,
                            );
                        }else{
                            println!("{}", err);
                        }
                    }
                }
            }
        }
    }
}
