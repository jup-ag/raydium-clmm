use super::full_math::MulDiv;
use super::liquidity_math;
use super::sqrt_price_math;
use crate::error::ErrorCode;
use crate::states::config::FEE_RATE_DENOMINATOR_VALUE;
use anchor_lang::prelude::*;
/// Result of a swap step
#[derive(Default, Debug)]
pub struct SwapStep {
    /// The price after swapping the amount in/out, not to exceed the price target
    pub sqrt_price_next_x64: u128,
    pub amount_in: u64,
    pub amount_out: u64,
    pub fee_amount: u64,
}

/// Optimized swap step computation
#[inline]
pub fn compute_swap_step(
    sqrt_price_current_x64: u128,
    sqrt_price_target_x64: u128,
    liquidity: u128,
    amount_remaining: u64,
    fee_rate: u32,
    is_base_input: bool,
    zero_for_one: bool,
    block_timestamp: u32,
) -> Result<SwapStep> {
    let mut swap_step = SwapStep::default();

    if is_base_input {
        // round up amount_in
        // In exact input case, amount_remaining is positive
        let amount_remaining_less_fee = (amount_remaining as u64)
            .mul_div_floor(
                (FEE_RATE_DENOMINATOR_VALUE - fee_rate).into(),
                u64::from(FEE_RATE_DENOMINATOR_VALUE),
            )
            .ok_or(ErrorCode::CalculateOverflow)?;

        // Try to calculate amount in range
        let amount_in_opt = calculate_amount_in_range(
            sqrt_price_current_x64,
            sqrt_price_target_x64,
            liquidity,
            zero_for_one,
            is_base_input,
            block_timestamp,
        )?;

        // Set next price based on whether we can reach target
        swap_step.sqrt_price_next_x64 = if let Some(amount_in) = amount_in_opt {
            if amount_remaining_less_fee >= amount_in {
                // Can reach target price, set the amount_in from initial calculation
                swap_step.amount_in = amount_in;
                sqrt_price_target_x64
            } else {
                sqrt_price_math::get_next_sqrt_price_from_input(
                    sqrt_price_current_x64,
                    liquidity,
                    amount_remaining_less_fee,
                    zero_for_one,
                )
            }
        } else {
            sqrt_price_math::get_next_sqrt_price_from_input(
                sqrt_price_current_x64,
                liquidity,
                amount_remaining_less_fee,
                zero_for_one,
            )
        };
    } else {
        // Exact output case
        let amount_out_opt = calculate_amount_in_range(
            sqrt_price_current_x64,
            sqrt_price_target_x64,
            liquidity,
            zero_for_one,
            is_base_input,
            block_timestamp,
        )?;

        swap_step.sqrt_price_next_x64 = if let Some(amount_out) = amount_out_opt {
            if amount_remaining >= amount_out {
                // Can reach target price, set the amount_out from initial calculation
                swap_step.amount_out = amount_out;
                sqrt_price_target_x64
            } else {
                sqrt_price_math::get_next_sqrt_price_from_output(
                    sqrt_price_current_x64,
                    liquidity,
                    amount_remaining,
                    zero_for_one,
                )
            }
        } else {
            sqrt_price_math::get_next_sqrt_price_from_output(
                sqrt_price_current_x64,
                liquidity,
                amount_remaining,
                zero_for_one,
            )
        };
    }

    // Check if we reached the target price
    let reached_target = sqrt_price_target_x64 == swap_step.sqrt_price_next_x64;

    // Calculate amounts based on actual price movement
    if zero_for_one {
        // Don't recalculate amount_in if we reached target with exact input
        if !(reached_target && is_base_input) {
            swap_step.amount_in = liquidity_math::get_delta_amount_0_unsigned(
                swap_step.sqrt_price_next_x64,
                sqrt_price_current_x64,
                liquidity,
                true,
            )?;
        }
        // Don't recalculate amount_out if we reached target with exact output
        if !(reached_target && !is_base_input) {
            swap_step.amount_out = liquidity_math::get_delta_amount_1_unsigned(
                swap_step.sqrt_price_next_x64,
                sqrt_price_current_x64,
                liquidity,
                false,
            )?;
        }
    } else {
        // Don't recalculate amount_in if we reached target with exact input
        if !(reached_target && is_base_input) {
            swap_step.amount_in = liquidity_math::get_delta_amount_1_unsigned(
                sqrt_price_current_x64,
                swap_step.sqrt_price_next_x64,
                liquidity,
                true,
            )?;
        }
        // Don't recalculate amount_out if we reached target with exact output
        if !(reached_target && !is_base_input) {
            swap_step.amount_out = liquidity_math::get_delta_amount_0_unsigned(
                sqrt_price_current_x64,
                swap_step.sqrt_price_next_x64,
                liquidity,
                false,
            )?;
        }
    }

    // For exact output case, cap the output amount if it exceeds requested
    if !is_base_input && swap_step.amount_out > amount_remaining {
        swap_step.amount_out = amount_remaining;
    }

    // Calculate fees - use original formula
    swap_step.fee_amount = if is_base_input && !reached_target {
        // we didn't reach the target, so take the remainder of the maximum input as fee
        // swap dust is granted as fee
        amount_remaining
            .checked_sub(swap_step.amount_in)
            .ok_or(ErrorCode::CalculateOverflow)?
    } else {
        // take pip percentage as fee
        swap_step
            .amount_in
            .mul_div_ceil(
                fee_rate.into(),
                (FEE_RATE_DENOMINATOR_VALUE - fee_rate).into(),
            )
            .ok_or(ErrorCode::CalculateOverflow)?
    };

    Ok(swap_step)
}

/// Pre calcumate amount_in or amount_out for the specified price range
/// The amount maybe overflow of u64 due to the `sqrt_price_target_x64` maybe unreasonable.
/// Therefore, this situation needs to be handled in `compute_swap_step` to recalculate the price that can be reached based on the amount.
#[cfg(not(test))]
fn calculate_amount_in_range(
    sqrt_price_current_x64: u128,
    sqrt_price_target_x64: u128,
    liquidity: u128,
    zero_for_one: bool,
    is_base_input: bool,
    _block_timestamp: u32,
) -> Result<Option<u64>> {
    if is_base_input {
        let result = if zero_for_one {
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
        };

        match result {
            Ok(value) => Ok(Some(value)),
            Err(err) if err == crate::error::ErrorCode::MaxTokenOverflow.into() => Ok(None),
            Err(_) => Err(ErrorCode::SqrtPriceLimitOverflow.into()),
        }
    } else {
        let result = if zero_for_one {
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
        };
        match result {
            Ok(value) => Ok(Some(value)),
            Err(err) if err == crate::error::ErrorCode::MaxTokenOverflow.into() => Ok(None),
            Err(_) => Err(ErrorCode::SqrtPriceLimitOverflow.into()),
        }
    }
}

#[cfg(test)]
fn calculate_amount_in_range(
    sqrt_price_current_x64: u128,
    sqrt_price_target_x64: u128,
    liquidity: u128,
    zero_for_one: bool,
    is_base_input: bool,
    block_timestamp: u32,
) -> Result<Option<u64>> {
    if is_base_input {
        let result = if zero_for_one {
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
        };

        if block_timestamp == 0 {
            if result.is_err() {
                return Err(ErrorCode::MaxTokenOverflow.into());
            } else {
                return Ok(Some(result?));
            }
        }
        if result.is_ok() {
            Ok(Some(result?))
        } else if result.err().unwrap() == crate::error::ErrorCode::MaxTokenOverflow.into() {
            return Ok(None);
        } else {
            return Err(ErrorCode::SqrtPriceLimitOverflow.into());
        }
    } else {
        let result = if zero_for_one {
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
        };
        if result.is_ok() || block_timestamp == 0 {
            Ok(Some(result?))
        } else if result.err().unwrap() == crate::error::ErrorCode::MaxTokenOverflow.into() {
            return Ok(None);
        } else {
            return Err(ErrorCode::SqrtPriceLimitOverflow.into());
        }
    }
}
#[cfg(test)]
mod swap_math_test {
    use crate::libraries::tick_math;

    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn compute_swap_step_test(
            sqrt_price_current_x64 in tick_math::MIN_SQRT_PRICE_X64..tick_math::MAX_SQRT_PRICE_X64,
            sqrt_price_target_x64 in tick_math::MIN_SQRT_PRICE_X64..tick_math::MAX_SQRT_PRICE_X64,
            liquidity in 1..u32::MAX as u128,
            amount_remaining in 1..u64::MAX,
            fee_rate in 1..FEE_RATE_DENOMINATOR_VALUE/2,
            is_base_input in proptest::bool::ANY,
        ) {
            prop_assume!(sqrt_price_current_x64 != sqrt_price_target_x64);

            let zero_for_one = sqrt_price_current_x64 > sqrt_price_target_x64;
            let swap_step = compute_swap_step(
                sqrt_price_current_x64,
                sqrt_price_target_x64,
                liquidity,
                amount_remaining,
                fee_rate,
                is_base_input,
                zero_for_one,
                1,
            )?;

            let amount_in = swap_step.amount_in;
            let amount_out = swap_step.amount_out;
            let sqrt_price_next_x64 = swap_step.sqrt_price_next_x64;
            let fee_amount = swap_step.fee_amount;

            let amount_used = if is_base_input {
                amount_in + fee_amount
            } else {
                amount_out
            };

            if sqrt_price_next_x64 != sqrt_price_target_x64 {
                assert!(amount_used == amount_remaining);
            } else {
                assert!(amount_used <= amount_remaining);
            }
            let price_lower = sqrt_price_current_x64.min(sqrt_price_target_x64);
            let price_upper = sqrt_price_current_x64.max(sqrt_price_target_x64);
            assert!(sqrt_price_next_x64 >= price_lower);
            assert!(sqrt_price_next_x64 <= price_upper);
        }
    }
}
