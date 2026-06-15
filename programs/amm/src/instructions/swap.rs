use crate::error::ErrorCode;
use crate::libraries::{
    big_num::U128, fixed_point_64, full_math::MulDiv, liquidity_math, swap_math, tick_math,
};
use crate::states::*;
use crate::util::*;
use anchor_lang::{prelude::*, solana_program};
use anchor_spl::token::{Token, TokenAccount};
use std::cell::RefMut;
use std::collections::{HashMap, VecDeque};
#[cfg(feature = "enable-log")]
use std::convert::identity;
use std::ops::Neg;

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
    #[account(mut)]
    pub input_token_account: Box<Account<'info, TokenAccount>>,

    /// The user token account for output token
    #[account(mut)]
    pub output_token_account: Box<Account<'info, TokenAccount>>,

    /// The vault token account for input token
    #[account(mut)]
    pub input_vault: Box<Account<'info, TokenAccount>>,

    /// The vault token account for output token
    #[account(mut)]
    pub output_vault: Box<Account<'info, TokenAccount>>,

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
    pub input_token_account: Box<Account<'info, TokenAccount>>,

    /// The user token account for output token
    pub output_token_account: Box<Account<'info, TokenAccount>>,

    /// The vault token account for input token
    pub input_vault: Box<Account<'info, TokenAccount>>,

    /// The vault token account for output token
    pub output_vault: Box<Account<'info, TokenAccount>>,

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
            state.tick_spacing_index =
                tick_spacing_index_from_tick(state.tick, state.tick_spacing)?;
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

    fn get_target_price_based_on_next_tick(
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
                    tick_spacing_index_from_tick(tick_index, self.tick_spacing)?;
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
        let fee_rate = if let Some(dynamic_fee_info) = &self.dynamic_fee_info {
            let dynamic_fee_rate =
                Self::compute_dynamic_fee_rate(dynamic_fee_info, self.tick_spacing)?;
            self.base_fee_rate
                .checked_add(dynamic_fee_rate)
                .ok_or(ErrorCode::CalculateOverflow)?
                .min(MAX_FEE_RATE_NUMERATOR)
        } else {
            // Use base fee if dynamic fee is disabled
            self.base_fee_rate
        };
        // The downstream fee math relies on `FEE_RATE_DENOMINATOR_VALUE - fee_rate` not underflowing.
        require!(
            fee_rate < FEE_RATE_DENOMINATOR_VALUE,
            ErrorCode::CalculateOverflow
        );
        Ok(fee_rate)
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
        // Widen before multiplying so a large `volatility_accumulator * tick_spacing` can't wrap u32.
        let crossed = u64::from(dynamic_fee_info.volatility_accumulator)
            .checked_mul(u64::from(tick_spacing))
            .ok_or(ErrorCode::CalculateOverflow)?;

        // Square the crossed value to create quadratic fee scaling
        let squared = crossed
            .checked_mul(crossed)
            .ok_or(ErrorCode::CalculateOverflow)?;

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

                if limit_order_result.amount_in != 0
                    || limit_order_result.amount_out != 0
                    || limit_order_result.amm_fee_amount != 0
                {
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
            state.update_dynamic_fee_index(zero_for_one, is_skipped_tick_spacing)?;
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
        }
        state.liquidity = liquidity_next;
    }

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
/// if is_base_input = true, return value is the max_amount_out, otherwise is min_amount_in
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
        //  x -> y, deposit x token from user to pool vault.
        transfer_from_user_to_pool_vault(
            &ctx.signer,
            &token_account_0.to_account_info(),
            &vault_0.to_account_info(),
            None,
            &ctx.token_program,
            None,
            swap_result.amount_0,
        )?;
        // x -> y，transfer y token from pool vault to user.
        transfer_from_pool_vault_to_user(
            &ctx.pool_state,
            &vault_1.to_account_info(),
            &token_account_1.to_account_info(),
            None,
            &ctx.token_program,
            None,
            swap_result.amount_1,
        )?;
    } else {
        transfer_from_user_to_pool_vault(
            &ctx.signer,
            &token_account_1.to_account_info(),
            &vault_1.to_account_info(),
            None,
            &ctx.token_program,
            None,
            swap_result.amount_1,
        )?;
        transfer_from_pool_vault_to_user(
            &ctx.pool_state,
            &vault_0.to_account_info(),
            &token_account_0.to_account_info(),
            None,
            &ctx.token_program,
            None,
            swap_result.amount_0,
        )?;
    }
    ctx.output_vault.reload()?;
    ctx.input_vault.reload()?;

    if zero_for_one {
        require_gte!(swap_price_before, swap_result.sqrt_price_x64);
    } else {
        require_gte!(swap_result.sqrt_price_x64, swap_price_before);
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
        output_balance_before
            .checked_sub(ctx.output_vault.amount)
            .ok_or(ErrorCode::CalculateOverflow.into())
    } else {
        ctx.input_vault
            .amount
            .checked_sub(input_balance_before)
            .ok_or(ErrorCode::CalculateOverflow.into())
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
///       * No `pool_state.update_reward_infos` and no `next_initialized_tick.cross()` —
///         the only useful output of `cross()` is `self.liquidity_net`, which is just a
///         field read. cross()'s side effects on `fee_growth_outside_*` /
///         `reward_growths_outside_x64` feed position accounting only, never consulted by
///         the swap math
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
    // Defend the quote path against malformed pool accounts. Every downstream tick-array /
    // dynamic-fee helper divides or mods by `tick_spacing`, and the swap math + tick_math
    // helpers assert valid `sqrt_price` / `tick_current` bounds — surface them as errors
    // instead of letting account-data poisoning reach a panic.
    require!(
        pool_state.tick_spacing != 0,
        ErrorCode::InvalidTickArrayBoundary
    );
    require!(
        pool_state.tick_current >= tick_math::MIN_TICK
            && pool_state.tick_current <= tick_math::MAX_TICK,
        ErrorCode::InvalidTickIndex
    );

    // Quote-path: skip `pool_state.update_reward_infos`
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
    // Quote-path: skip `require_keys_eq!(tick_array_current.pool_id, pool_state.key())`
    require_eq!(
        tick_array_current.start_tick_index,
        current_valid_tick_array_start_index,
        ErrorCode::InvalidFirstTickArrayAccount
    );

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

                while tick_array_current.start_tick_index != next_tick_array_index {
                    tick_array_current = tick_array_states
                        .pop_front()
                        .ok_or(ErrorCode::NotEnoughTickArrayAccount)?;
                    // Quote-path: skip `require_keys_eq!(tick_array_current.pool_id, pool_state.key())`.
                }
                current_valid_tick_array_start_index = next_tick_array_index;

                *tick_array_current.first_initialized_tick(zero_for_one)?
            }
        };
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

                if limit_order_result.amount_in != 0
                    || limit_order_result.amount_out != 0
                    || limit_order_result.amm_fee_amount != 0
                {
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
                    // Quote-path: skip `next_initialized_tick.cross(fee_growth_0, fee_growth_1,
                    // &reward_infos)` — the only useful return is `self.liquidity_net`, which is
                    // just read directly. cross()'s side effects are for position fee/reward accounting,
                    // which the quote path does not need.
                    let mut liquidity_net = next_initialized_tick.liquidity_net;
                    if zero_for_one {
                        liquidity_net = liquidity_net.neg();
                    }
                    liquidity_next = liquidity_math::add_delta(state.liquidity, liquidity_net)?;
                }

                // Quote-path: skip `tick_array_current.update_tick_state(...)` (mutates the
                // tick array; not needed since `next_initialized_tick` is not revisited in this quote)

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
            state.update_dynamic_fee_index(zero_for_one, is_skipped_tick_spacing)?;
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
        }
        state.liquidity = liquidity_next;
    }

    // Quote-path: skip `observation_state.update(...)` and `pool_state.update_after_swap(...)`.

    let (amount_0, amount_1, _trade_fee_0, _trade_fee_1) = state.settle_amounts(
        amount_specified,
        zero_for_one,
        is_base_input,
        pool_state.is_fee_on_token0(zero_for_one),
    )?;

    Ok((state, amount_0, amount_1))
}
