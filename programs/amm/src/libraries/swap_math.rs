use super::full_math::MulDiv;
use super::liquidity_math;
use super::sqrt_price_math;
use crate::error::ErrorCode;
use crate::states::config::FEE_RATE_DENOMINATOR_VALUE;
use anchor_lang::prelude::*;

/// Result of a swap computation
///
/// Contains the computed price, amounts, and fees after executing a swap calculation.
#[derive(Default, Debug)]
pub struct SwapComputationResult {
    /// The price after swapping the amount in/out, not to exceed the price target
    pub sqrt_price_next_x64: u128,
    pub amount_in: u64,
    pub amount_out: u64,
    pub fee_amount: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct SwapStepAmountCache {
    pub sqrt_price_current_x64: u128,
    pub sqrt_price_target_x64: u128,
    pub liquidity: u128,
    pub zero_for_one: bool,
    pub amount_in: Option<u64>,
    pub amount_out: Option<u64>,
}

impl SwapStepAmountCache {
    #[inline(always)]
    fn matches(
        self,
        sqrt_price_current_x64: u128,
        sqrt_price_target_x64: u128,
        liquidity: u128,
        zero_for_one: bool,
    ) -> bool {
        self.sqrt_price_current_x64 == sqrt_price_current_x64
            && self.sqrt_price_target_x64 == sqrt_price_target_x64
            && self.liquidity == liquidity
            && self.zero_for_one == zero_for_one
    }

    #[inline(always)]
    fn amount_in(
        self,
        sqrt_price_current_x64: u128,
        sqrt_price_target_x64: u128,
        liquidity: u128,
        zero_for_one: bool,
    ) -> Option<Option<u64>> {
        self.matches(
            sqrt_price_current_x64,
            sqrt_price_target_x64,
            liquidity,
            zero_for_one,
        )
        .then_some(self.amount_in)
    }

    #[inline(always)]
    fn amount_out(
        self,
        sqrt_price_current_x64: u128,
        sqrt_price_target_x64: u128,
        liquidity: u128,
        zero_for_one: bool,
    ) -> Option<Option<u64>> {
        self.matches(
            sqrt_price_current_x64,
            sqrt_price_target_x64,
            liquidity,
            zero_for_one,
        )
        .then_some(self.amount_out)
    }
}

pub fn cached_swap_step_amounts(
    sqrt_price_current_x64: u128,
    sqrt_price_target_x64: u128,
    liquidity: u128,
    zero_for_one: bool,
) -> Result<SwapStepAmountCache> {
    Ok(SwapStepAmountCache {
        sqrt_price_current_x64,
        sqrt_price_target_x64,
        liquidity,
        zero_for_one,
        amount_in: calculate_amount_in_range(
            sqrt_price_current_x64,
            sqrt_price_target_x64,
            liquidity,
            zero_for_one,
            true,
        )?,
        amount_out: calculate_amount_in_range(
            sqrt_price_current_x64,
            sqrt_price_target_x64,
            liquidity,
            zero_for_one,
            false,
        )?,
    })
}

impl SwapComputationResult {
    pub fn new(sqrt_price_next_x64: u128) -> Self {
        Self {
            sqrt_price_next_x64,
            amount_in: 0,
            amount_out: 0,
            fee_amount: 0,
        }
    }
}

/// Computes the result of swapping some amount in, or amount out, given the parameters of the swap
///
/// `fee_rate` is the effective per-step rate the caller has already resolved (e.g. base + dynamic).
/// `is_fee_on_input` selects which side the fee is collected from, per `PoolState::is_fee_on_input`.
pub fn compute_swap(
    sqrt_price_current_x64: u128,
    sqrt_price_target_x64: u128,
    liquidity: u128,
    amount_remaining: u64,
    fee_rate: u32,
    is_base_input: bool,
    zero_for_one: bool,
    is_fee_on_input: bool,
) -> Result<SwapComputationResult> {
    compute_swap_with_cached_amounts(
        sqrt_price_current_x64,
        sqrt_price_target_x64,
        liquidity,
        amount_remaining,
        fee_rate,
        is_base_input,
        zero_for_one,
        is_fee_on_input,
        None,
    )
}

pub fn compute_swap_with_cached_amounts(
    sqrt_price_current_x64: u128,
    sqrt_price_target_x64: u128,
    liquidity: u128,
    amount_remaining: u64,
    fee_rate: u32,
    is_base_input: bool,
    zero_for_one: bool,
    is_fee_on_input: bool,
    cached_amounts: Option<SwapStepAmountCache>,
) -> Result<SwapComputationResult> {
    let mut result = SwapComputationResult::default();
    if is_base_input {
        let amount_for_price_calc = if is_fee_on_input {
            // Fee from input: amount_remaining includes fee, deduct fee first
            amount_remaining
                .mul_div_floor(
                    (FEE_RATE_DENOMINATOR_VALUE - fee_rate).into(),
                    u64::from(FEE_RATE_DENOMINATOR_VALUE),
                )
                .ok_or(ErrorCode::CalculateOverflow)?
        } else {
            amount_remaining
        };

        let amount_in = if let Some(amount_in) = cached_amounts.and_then(|cached| {
            cached.amount_in(
                sqrt_price_current_x64,
                sqrt_price_target_x64,
                liquidity,
                zero_for_one,
            )
        }) {
            amount_in
        } else {
            calculate_amount_in_range(
                sqrt_price_current_x64,
                sqrt_price_target_x64,
                liquidity,
                zero_for_one,
                is_base_input,
            )?
        };
        if let Some(v) = amount_in {
            result.amount_in = v;
        }

        result.sqrt_price_next_x64 =
            if amount_in.is_some() && amount_for_price_calc >= result.amount_in {
                sqrt_price_target_x64
            } else {
                sqrt_price_math::get_next_sqrt_price_from_input(
                    sqrt_price_current_x64,
                    liquidity,
                    amount_for_price_calc,
                    zero_for_one,
                )?
            };
    } else {
        // amount_remaining is the net output the user wants to receive (after fee deduction if fee is from output)
        let amount_for_price_calc = if is_fee_on_input {
            amount_remaining
        } else {
            // Fee from output: amount_remaining is net output, we need gross output for price calculation
            // gross_output = net_output / (1 - fee_rate / FEE_RATE_DENOMINATOR)
            amount_remaining
                .mul_div_ceil(
                    u64::from(FEE_RATE_DENOMINATOR_VALUE).into(),
                    (FEE_RATE_DENOMINATOR_VALUE - fee_rate).into(),
                )
                .ok_or(ErrorCode::CalculateOverflow)?
        };

        let amount_out = if let Some(amount_out) = cached_amounts.and_then(|cached| {
            cached.amount_out(
                sqrt_price_current_x64,
                sqrt_price_target_x64,
                liquidity,
                zero_for_one,
            )
        }) {
            amount_out
        } else {
            calculate_amount_in_range(
                sqrt_price_current_x64,
                sqrt_price_target_x64,
                liquidity,
                zero_for_one,
                is_base_input,
            )?
        };
        if let Some(v) = amount_out {
            result.amount_out = v;
        }
        result.sqrt_price_next_x64 =
            if amount_out.is_some() && amount_for_price_calc >= result.amount_out {
                sqrt_price_target_x64
            } else {
                sqrt_price_math::get_next_sqrt_price_from_output(
                    sqrt_price_current_x64,
                    liquidity,
                    amount_for_price_calc,
                    zero_for_one,
                )?
            }
    }

    if zero_for_one {
        require_gte!(result.sqrt_price_next_x64, sqrt_price_target_x64);
    } else {
        require_gte!(sqrt_price_target_x64, result.sqrt_price_next_x64);
    }

    // whether we reached the max possible price for the given ticks
    let max = sqrt_price_target_x64 == result.sqrt_price_next_x64;
    // get the input / output amounts when target price is not reached
    if zero_for_one {
        // if max is reached for exact input case, entire amount_in is needed
        if !(max && is_base_input) {
            result.amount_in = if max {
                cached_amounts
                    .and_then(|cached| {
                        cached
                            .amount_in(
                                sqrt_price_current_x64,
                                sqrt_price_target_x64,
                                liquidity,
                                zero_for_one,
                            )
                            .flatten()
                    })
                    .map(Ok)
                    .unwrap_or_else(|| {
                        liquidity_math::get_delta_amount_0_unsigned(
                            result.sqrt_price_next_x64,
                            sqrt_price_current_x64,
                            liquidity,
                            true,
                        )
                    })?
            } else {
                liquidity_math::get_delta_amount_0_unsigned(
                    result.sqrt_price_next_x64,
                    sqrt_price_current_x64,
                    liquidity,
                    true,
                )?
            }
        };
        // if max is reached for exact output case, entire amount_out is needed
        if !(max && !is_base_input) {
            result.amount_out = if max {
                cached_amounts
                    .and_then(|cached| {
                        cached
                            .amount_out(
                                sqrt_price_current_x64,
                                sqrt_price_target_x64,
                                liquidity,
                                zero_for_one,
                            )
                            .flatten()
                    })
                    .map(Ok)
                    .unwrap_or_else(|| {
                        liquidity_math::get_delta_amount_1_unsigned(
                            result.sqrt_price_next_x64,
                            sqrt_price_current_x64,
                            liquidity,
                            false,
                        )
                    })?
            } else {
                liquidity_math::get_delta_amount_1_unsigned(
                    result.sqrt_price_next_x64,
                    sqrt_price_current_x64,
                    liquidity,
                    false,
                )?
            };
        };
    } else {
        if !(max && is_base_input) {
            result.amount_in = if max {
                cached_amounts
                    .and_then(|cached| {
                        cached
                            .amount_in(
                                sqrt_price_current_x64,
                                sqrt_price_target_x64,
                                liquidity,
                                zero_for_one,
                            )
                            .flatten()
                    })
                    .map(Ok)
                    .unwrap_or_else(|| {
                        liquidity_math::get_delta_amount_1_unsigned(
                            sqrt_price_current_x64,
                            result.sqrt_price_next_x64,
                            liquidity,
                            true,
                        )
                    })?
            } else {
                liquidity_math::get_delta_amount_1_unsigned(
                    sqrt_price_current_x64,
                    result.sqrt_price_next_x64,
                    liquidity,
                    true,
                )?
            }
        };
        if !(max && !is_base_input) {
            result.amount_out = if max {
                cached_amounts
                    .and_then(|cached| {
                        cached
                            .amount_out(
                                sqrt_price_current_x64,
                                sqrt_price_target_x64,
                                liquidity,
                                zero_for_one,
                            )
                            .flatten()
                    })
                    .map(Ok)
                    .unwrap_or_else(|| {
                        liquidity_math::get_delta_amount_0_unsigned(
                            sqrt_price_current_x64,
                            result.sqrt_price_next_x64,
                            liquidity,
                            false,
                        )
                    })?
            } else {
                liquidity_math::get_delta_amount_0_unsigned(
                    sqrt_price_current_x64,
                    result.sqrt_price_next_x64,
                    liquidity,
                    false,
                )?
            }
        };
    }

    if is_base_input {
        if is_fee_on_input {
            if result.sqrt_price_next_x64 != sqrt_price_target_x64 {
                result.fee_amount = amount_remaining
                    .checked_sub(result.amount_in)
                    .ok_or(ErrorCode::CalculateOverflow)?;
            } else {
                result.fee_amount = result
                    .amount_in
                    .mul_div_ceil(
                        fee_rate.into(),
                        (FEE_RATE_DENOMINATOR_VALUE - fee_rate).into(),
                    )
                    .ok_or(ErrorCode::CalculateOverflow)?;
            }
        } else {
            // Fee from output: result.amount_out is gross output, fee is calculated from gross output
            // fee = gross_output * fee_rate / FEE_RATE_DENOMINATOR
            result.fee_amount = result
                .amount_out
                .mul_div_ceil(fee_rate.into(), FEE_RATE_DENOMINATOR_VALUE.into())
                .ok_or(ErrorCode::CalculateOverflow)?;
            // Deduct fee from output: user receives net output
            result.amount_out = result
                .amount_out
                .checked_sub(result.fee_amount)
                .ok_or(ErrorCode::CalculateOverflow)?;

            // Partial step: the price moved less than the exact input warrants (rounded toward the
            // pool — down for one_for_zero, up for zero_for_one), so amount_in recomputed from that
            // move can be below the available input, leaving an un-tradeable dust (< liquidity/Q64).
            // Fee-on-input folds it into the fee, fee-on-output cannot, so it would stall the loop.
            // Charge the full input (== amount_remaining here); the sub-unit excess goes to the pool.
            if !max {
                result.amount_in = amount_remaining;
            }
        }
    } else {
        if is_fee_on_input {
            // Fee from input: amount_remaining is the desired gross output
            // Cap the gross output amount to the remaining amount
            result.amount_out = result.amount_out.min(amount_remaining);
            result.fee_amount = result
                .amount_in
                .mul_div_ceil(
                    fee_rate.into(),
                    (FEE_RATE_DENOMINATOR_VALUE - fee_rate).into(),
                )
                .ok_or(ErrorCode::CalculateOverflow)?;
        } else {
            result.fee_amount = result
                .amount_out
                .mul_div_ceil(fee_rate.into(), FEE_RATE_DENOMINATOR_VALUE.into())
                .ok_or(ErrorCode::CalculateOverflow)?;

            // Calculate net output
            let net_output = result
                .amount_out
                .checked_sub(result.fee_amount)
                .ok_or(ErrorCode::CalculateOverflow)?;

            // Cap net output to amount_remaining (user's desired net output)
            // If net output exceeds amount_remaining, adjust fee to cap it
            if net_output > amount_remaining {
                // Adjust fee so that net output = amount_remaining
                result.fee_amount = result
                    .amount_out
                    .checked_sub(amount_remaining)
                    .ok_or(ErrorCode::CalculateOverflow)?;
                result.amount_out = amount_remaining;
            } else {
                // Deduct fee from output: user receives net output
                result.amount_out = net_output;
            }
        }
    }

    // Dust stall guard: the caller's swap loop only exits when either
    // `amount_specified_remaining` drains or `sqrt_price_x64` reaches the
    // target/limit. If neither moves this step (amount-side decrement is 0
    // AND price is unchanged), the loop spins until CU exhaustion. Zero-amount
    // steps with price movement (e.g. traversing an empty-liquidity gap) are
    // still valid and pass through.
    let progress = if is_base_input {
        if is_fee_on_input {
            result
                .amount_in
                .checked_add(result.fee_amount)
                .ok_or(ErrorCode::CalculateOverflow)?
        } else {
            result.amount_in
        }
    } else {
        result.amount_out
    };
    if progress == 0 && result.sqrt_price_next_x64 == sqrt_price_current_x64 {
        return Err(ErrorCode::LiquidityInsufficient.into());
    }

    Ok(result)
}

/// Pre-calculate amount_in or amount_out for the specified price range.
///
/// The amount may overflow u64 if `sqrt_price_target_x64` is unreasonable;
/// in that case `Ok(None)` is returned so the caller can clamp the target.
fn calculate_amount_in_range(
    sqrt_price_current_x64: u128,
    sqrt_price_target_x64: u128,
    liquidity: u128,
    zero_for_one: bool,
    is_base_input: bool,
) -> Result<Option<u64>> {
    let result = if is_base_input {
        if zero_for_one {
            liquidity_math::get_delta_amount_0_unsigned(
                sqrt_price_target_x64,
                sqrt_price_current_x64,
                liquidity,
                true,
            )
        } else {
            liquidity_math::get_delta_amount_1_unsigned(
                sqrt_price_current_x64,
                sqrt_price_target_x64,
                liquidity,
                true,
            )
        }
    } else {
        if zero_for_one {
            liquidity_math::get_delta_amount_1_unsigned(
                sqrt_price_target_x64,
                sqrt_price_current_x64,
                liquidity,
                false,
            )
        } else {
            liquidity_math::get_delta_amount_0_unsigned(
                sqrt_price_current_x64,
                sqrt_price_target_x64,
                liquidity,
                false,
            )
        }
    };

    match result {
        Ok(v) => Ok(Some(v)),
        Err(e) if e == ErrorCode::MaxTokenOverflow.into() => Ok(None),
        Err(_) => Err(ErrorCode::SqrtPriceLimitOverflow.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::libraries::tick_math;

    fn assert_same_swap(
        sqrt_price_current_x64: u128,
        sqrt_price_target_x64: u128,
        liquidity: u128,
        amount_remaining: u64,
        fee_rate: u32,
        is_base_input: bool,
        zero_for_one: bool,
        is_fee_on_input: bool,
    ) -> Result<()> {
        let cached_amounts = cached_swap_step_amounts(
            sqrt_price_current_x64,
            sqrt_price_target_x64,
            liquidity,
            zero_for_one,
        )?;
        let uncached = compute_swap(
            sqrt_price_current_x64,
            sqrt_price_target_x64,
            liquidity,
            amount_remaining,
            fee_rate,
            is_base_input,
            zero_for_one,
            is_fee_on_input,
        )?;
        let cached = compute_swap_with_cached_amounts(
            sqrt_price_current_x64,
            sqrt_price_target_x64,
            liquidity,
            amount_remaining,
            fee_rate,
            is_base_input,
            zero_for_one,
            is_fee_on_input,
            Some(cached_amounts),
        )?;

        assert_eq!(uncached.sqrt_price_next_x64, cached.sqrt_price_next_x64);
        assert_eq!(uncached.amount_in, cached.amount_in);
        assert_eq!(uncached.amount_out, cached.amount_out);
        assert_eq!(uncached.fee_amount, cached.fee_amount);
        Ok(())
    }

    #[test]
    fn cached_swap_step_amounts_match_uncached_compute_swap() -> Result<()> {
        let liquidity = 1_000_000_000_000_000u128;
        let amount_remaining = 1_000_000_000_000_000u64;
        let fee_rate = 2_500u32;

        for zero_for_one in [true, false] {
            let (sqrt_price_current_x64, sqrt_price_target_x64) = if zero_for_one {
                (
                    tick_math::get_sqrt_price_at_tick(100)?,
                    tick_math::get_sqrt_price_at_tick(0)?,
                )
            } else {
                (
                    tick_math::get_sqrt_price_at_tick(0)?,
                    tick_math::get_sqrt_price_at_tick(100)?,
                )
            };

            for is_base_input in [true, false] {
                for is_fee_on_input in [true, false] {
                    assert_same_swap(
                        sqrt_price_current_x64,
                        sqrt_price_target_x64,
                        liquidity,
                        amount_remaining,
                        fee_rate,
                        is_base_input,
                        zero_for_one,
                        is_fee_on_input,
                    )?;
                }
            }
        }

        Ok(())
    }
}
