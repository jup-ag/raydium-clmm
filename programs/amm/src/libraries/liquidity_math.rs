use super::big_num::U128;
use super::big_num::U256;
use super::fixed_point_64;
use super::full_math::MulDiv;
use super::tick_math;
use super::unsafe_math::UnsafeMathTrait;
use crate::error::ErrorCode;
use anchor_lang::prelude::*;

/// Add a signed liquidity delta to liquidity and revert if it overflows or underflows
///
/// # Arguments
///
/// * `x` - The liquidity (L) before change
/// * `y` - The delta (ΔL) by which liquidity should be changed
///
pub fn add_delta(x: u128, y: i128) -> Result<u128> {
    let z: u128;
    if y < 0 {
        z = x - u128::try_from(-y).map_err(|_| ErrorCode::CalculateOverflow)?;
        require_gt!(x, z, ErrorCode::LiquiditySubValueErr);
    } else {
        z = x + u128::try_from(y).map_err(|_| ErrorCode::CalculateOverflow)?;
        require_gte!(z, x, ErrorCode::LiquidityAddValueErr);
    }

    Ok(z)
}

/// Computes the amount of liquidity received for a given amount of token_0 and price range
/// Calculates ΔL = Δx (√P_upper x √P_lower)/(√P_upper - √P_lower)
pub fn get_liquidity_from_amount_0(
    mut sqrt_ratio_a_x64: u128,
    mut sqrt_ratio_b_x64: u128,
    amount_0: u64,
) -> Result<u128> {
    // sqrt_ratio_a_x64 should hold the smaller value
    if sqrt_ratio_a_x64 > sqrt_ratio_b_x64 {
        std::mem::swap(&mut sqrt_ratio_a_x64, &mut sqrt_ratio_b_x64);
    };
    let intermediate = U128::from(sqrt_ratio_a_x64)
        .mul_div_floor(
            U128::from(sqrt_ratio_b_x64),
            U128::from(fixed_point_64::Q64),
        )
        .ok_or(ErrorCode::CalculateOverflow)?;

    Ok(U128::from(amount_0)
        .mul_div_floor(
            intermediate,
            U128::from(sqrt_ratio_b_x64 - sqrt_ratio_a_x64),
        )
        .ok_or(ErrorCode::CalculateOverflow)?
        .as_u128())
}

/// Computes the amount of liquidity received for a given amount of token_1 and price range
/// Calculates ΔL = Δy / (√P_upper - √P_lower)
pub fn get_liquidity_from_amount_1(
    mut sqrt_ratio_a_x64: u128,
    mut sqrt_ratio_b_x64: u128,
    amount_1: u64,
) -> Result<u128> {
    // sqrt_ratio_a_x64 should hold the smaller value
    if sqrt_ratio_a_x64 > sqrt_ratio_b_x64 {
        std::mem::swap(&mut sqrt_ratio_a_x64, &mut sqrt_ratio_b_x64);
    };

    Ok(U128::from(amount_1)
        .mul_div_floor(
            U128::from(fixed_point_64::Q64),
            U128::from(sqrt_ratio_b_x64 - sqrt_ratio_a_x64),
        )
        .ok_or(ErrorCode::CalculateOverflow)?
        .as_u128())
}

/// Computes the maximum amount of liquidity received for a given amount of token_0, token_1, the current
/// pool prices and the prices at the tick boundaries
pub fn get_liquidity_from_amounts(
    sqrt_ratio_x64: u128,
    mut sqrt_ratio_a_x64: u128,
    mut sqrt_ratio_b_x64: u128,
    amount_0: u64,
    amount_1: u64,
) -> Result<u128> {
    // sqrt_ratio_a_x64 should hold the smaller value
    if sqrt_ratio_a_x64 > sqrt_ratio_b_x64 {
        std::mem::swap(&mut sqrt_ratio_a_x64, &mut sqrt_ratio_b_x64);
    };

    Ok(if sqrt_ratio_x64 <= sqrt_ratio_a_x64 {
        // If P ≤ P_lower, only token_0 liquidity is active
        get_liquidity_from_amount_0(sqrt_ratio_a_x64, sqrt_ratio_b_x64, amount_0)?
    } else if sqrt_ratio_x64 < sqrt_ratio_b_x64 {
        // If P_lower < P < P_upper, active liquidity is the minimum of the liquidity provided
        // by token_0 and token_1
        u128::min(
            get_liquidity_from_amount_0(sqrt_ratio_x64, sqrt_ratio_b_x64, amount_0)?,
            get_liquidity_from_amount_1(sqrt_ratio_a_x64, sqrt_ratio_x64, amount_1)?,
        )
    } else {
        // If P ≥ P_upper, only token_1 liquidity is active
        get_liquidity_from_amount_1(sqrt_ratio_a_x64, sqrt_ratio_b_x64, amount_1)?
    })
}

/// Computes the maximum amount of liquidity received for a given amount of token_0, token_1, the current
/// pool prices and the prices at the tick boundaries
pub fn get_liquidity_from_single_amount_0(
    sqrt_ratio_x64: u128,
    mut sqrt_ratio_a_x64: u128,
    mut sqrt_ratio_b_x64: u128,
    amount_0: u64,
) -> Result<u128> {
    // sqrt_ratio_a_x64 should hold the smaller value
    if sqrt_ratio_a_x64 > sqrt_ratio_b_x64 {
        std::mem::swap(&mut sqrt_ratio_a_x64, &mut sqrt_ratio_b_x64);
    };

    Ok(if sqrt_ratio_x64 <= sqrt_ratio_a_x64 {
        // If P ≤ P_lower, only token_0 liquidity is active
        get_liquidity_from_amount_0(sqrt_ratio_a_x64, sqrt_ratio_b_x64, amount_0)?
    } else if sqrt_ratio_x64 < sqrt_ratio_b_x64 {
        // If P_lower < P < P_upper, active liquidity is the minimum of the liquidity provided
        // by token_0 and token_1
        get_liquidity_from_amount_0(sqrt_ratio_x64, sqrt_ratio_b_x64, amount_0)?
    } else {
        // If P ≥ P_upper, only token_1 liquidity is active
        0
    })
}

/// Computes the maximum amount of liquidity received for a given amount of token_0, token_1, the current
/// pool prices and the prices at the tick boundaries
pub fn get_liquidity_from_single_amount_1(
    sqrt_ratio_x64: u128,
    mut sqrt_ratio_a_x64: u128,
    mut sqrt_ratio_b_x64: u128,
    amount_1: u64,
) -> Result<u128> {
    // sqrt_ratio_a_x64 should hold the smaller value
    if sqrt_ratio_a_x64 > sqrt_ratio_b_x64 {
        std::mem::swap(&mut sqrt_ratio_a_x64, &mut sqrt_ratio_b_x64);
    };

    Ok(if sqrt_ratio_x64 <= sqrt_ratio_a_x64 {
        // If P ≤ P_lower, only token_0 liquidity is active
        0
    } else if sqrt_ratio_x64 < sqrt_ratio_b_x64 {
        // If P_lower < P < P_upper, active liquidity is the minimum of the liquidity provided
        // by token_0 and token_1
        get_liquidity_from_amount_1(sqrt_ratio_a_x64, sqrt_ratio_x64, amount_1)?
    } else {
        // If P ≥ P_upper, only token_1 liquidity is active
        get_liquidity_from_amount_1(sqrt_ratio_a_x64, sqrt_ratio_b_x64, amount_1)?
    })
}

/// Optimized delta amount_0 calculation
/// Formula: Δx = L * (√P_upper - √P_lower) / (√P_upper * √P_lower)
#[inline]
pub fn get_delta_amount_0_unsigned(
    mut sqrt_ratio_a_x64: u128,
    mut sqrt_ratio_b_x64: u128,
    liquidity: u128,
    round_up: bool,
) -> Result<u64> {
    // Ensure sqrt_ratio_a_x64 is smaller
    if sqrt_ratio_a_x64 > sqrt_ratio_b_x64 {
        std::mem::swap(&mut sqrt_ratio_a_x64, &mut sqrt_ratio_b_x64);
    }

    // Early return for zero case
    if sqrt_ratio_a_x64 == 0 || liquidity == 0 {
        return Ok(0);
    }

    // Ultra-fast path for very small values (nanosecond performance)
    if liquidity <= 65536 && sqrt_ratio_a_x64 <= 1000000 && sqrt_ratio_b_x64 <= 1000000 {
        let price_diff = sqrt_ratio_b_x64 - sqrt_ratio_a_x64;
        
        // Use checked arithmetic even in fast path to prevent overflow
        if let Some(numerator) = liquidity.checked_shl(64) {
            if let Some(denominator) = sqrt_ratio_a_x64.checked_mul(sqrt_ratio_b_x64) {
                if denominator > 0 {
                    if let Some(product) = numerator.checked_mul(price_diff) {
                        let result = product / denominator;
                        if result <= u64::MAX as u128 {
                            return Ok(result as u64);
                        }
                    }
                }
            }
        }
    }

    // Fast path for larger values that fit in u128 arithmetic
    if liquidity <= u64::MAX as u128 && 
       sqrt_ratio_a_x64 <= u64::MAX as u128 && 
       sqrt_ratio_b_x64 <= u64::MAX as u128 {
        
        let price_diff = sqrt_ratio_b_x64 - sqrt_ratio_a_x64;
        
        // Check for overflow in the numerator calculation
        if let Some(numerator) = liquidity.checked_shl(64) {
            // Check for overflow in the multiplication
            if let Some(num_times_diff) = numerator.checked_mul(price_diff) {
                if let Some(denominator) = sqrt_ratio_a_x64.checked_mul(sqrt_ratio_b_x64) {
                    if denominator > 0 {
                        let result = if round_up {
                            num_times_diff.div_ceil(denominator)
                        } else {
                            num_times_diff / denominator
                        };
                        
                        if result <= u64::MAX as u128 {
                            return Ok(result as u64);
                        }
                    }
                }
            }
        }
    }

    // Fallback to U256 arithmetic for larger values
    let numerator_1 = U256::from(liquidity) << fixed_point_64::RESOLUTION;
    let numerator_2 = U256::from(sqrt_ratio_b_x64 - sqrt_ratio_a_x64);
    let denominator = U256::from(sqrt_ratio_a_x64) * U256::from(sqrt_ratio_b_x64);

    let result = if round_up {
        U256::div_rounding_up(numerator_1 * numerator_2, denominator)
    } else {
        (numerator_1 * numerator_2) / denominator
    };

    if result > U256::from(u64::MAX) {
        return Err(ErrorCode::MaxTokenOverflow.into());
    }
    Ok(result.as_u64())
}

/// Optimized delta amount_1 calculation  
/// Formula: Δy = L * (√P_upper - √P_lower)
#[inline]
pub fn get_delta_amount_1_unsigned(
    mut sqrt_ratio_a_x64: u128,
    mut sqrt_ratio_b_x64: u128,
    liquidity: u128,
    round_up: bool,
) -> Result<u64> {
    // Ensure sqrt_ratio_a_x64 is smaller
    if sqrt_ratio_a_x64 > sqrt_ratio_b_x64 {
        std::mem::swap(&mut sqrt_ratio_a_x64, &mut sqrt_ratio_b_x64);
    }

    // Early return for zero case
    if liquidity == 0 || sqrt_ratio_a_x64 == sqrt_ratio_b_x64 {
        return Ok(0);
    }

    let price_diff = sqrt_ratio_b_x64 - sqrt_ratio_a_x64;
    
    // Ultra-fast path for small values (nanosecond performance)
    if liquidity <= u64::MAX as u128 && price_diff <= u64::MAX as u128 {
        // Use checked arithmetic to prevent overflow
        if let Some(product) = liquidity.checked_mul(price_diff) {
            let result = if round_up {
                // For ceiling division: (a + b - 1) / b becomes (a + b - 1) >> log2(b)
                if let Some(sum) = product.checked_add((1u128 << 64) - 1) {
                    sum >> 64
                } else {
                    // Fallback to U256 if overflow
                    u64::MAX as u128 + 1 // This will trigger the fallback below
                }
            } else {
                product >> 64  // Direct bit shift instead of division
            };
            
            if result <= u64::MAX as u128 {
                return Ok(result as u64);
            }
        }
    }
    
    // Fallback to U256 arithmetic for larger values
    let result = if round_up {
        U256::from(liquidity).mul_div_ceil(
            U256::from(price_diff),
            U256::from(fixed_point_64::Q64),
        )
    } else {
        U256::from(liquidity).mul_div_floor(
            U256::from(price_diff),
            U256::from(fixed_point_64::Q64),
        )
    }
    .ok_or(ErrorCode::CalculateOverflow)?;

    if result > U256::from(u64::MAX) {
        return Err(ErrorCode::MaxTokenOverflow.into());
    }
    Ok(result.as_u64())
}

/// Helper function to get signed delta amount_0 for given liquidity and price range
pub fn get_delta_amount_0_signed(
    sqrt_ratio_a_x64: u128,
    sqrt_ratio_b_x64: u128,
    liquidity: i128,
) -> Result<u64> {
    if liquidity < 0 {
        get_delta_amount_0_unsigned(
            sqrt_ratio_a_x64,
            sqrt_ratio_b_x64,
            u128::try_from(-liquidity).map_err(|_| ErrorCode::CalculateOverflow)?,
            false,
        )
    } else {
        get_delta_amount_0_unsigned(
            sqrt_ratio_a_x64,
            sqrt_ratio_b_x64,
            u128::try_from(liquidity).map_err(|_| ErrorCode::CalculateOverflow)?,
            true,
        )
    }
}

/// Helper function to get signed delta amount_1 for given liquidity and price range
pub fn get_delta_amount_1_signed(
    sqrt_ratio_a_x64: u128,
    sqrt_ratio_b_x64: u128,
    liquidity: i128,
) -> Result<u64> {
    if liquidity < 0 {
        get_delta_amount_1_unsigned(
            sqrt_ratio_a_x64,
            sqrt_ratio_b_x64,
            u128::try_from(-liquidity).map_err(|_| ErrorCode::CalculateOverflow)?,
            false,
        )
    } else {
        get_delta_amount_1_unsigned(
            sqrt_ratio_a_x64,
            sqrt_ratio_b_x64,
            u128::try_from(liquidity).map_err(|_| ErrorCode::CalculateOverflow)?,
            true,
        )
    }
}

pub fn get_delta_amounts_signed(
    tick_current: i32,
    sqrt_price_x64_current: u128,
    tick_lower: i32,
    tick_upper: i32,
    liquidity_delta: i128,
) -> Result<(u64, u64)> {
    let mut amount_0 = 0;
    let mut amount_1 = 0;
    if tick_current < tick_lower {
        amount_0 = get_delta_amount_0_signed(
            tick_math::get_sqrt_price_at_tick(tick_lower)?,
            tick_math::get_sqrt_price_at_tick(tick_upper)?,
            liquidity_delta,
        )?;
    } else if tick_current < tick_upper {
        amount_0 = get_delta_amount_0_signed(
            sqrt_price_x64_current,
            tick_math::get_sqrt_price_at_tick(tick_upper)?,
            liquidity_delta,
        )?;
        amount_1 = get_delta_amount_1_signed(
            tick_math::get_sqrt_price_at_tick(tick_lower)?,
            sqrt_price_x64_current,
            liquidity_delta,
        )?;
    } else {
        amount_1 = get_delta_amount_1_signed(
            tick_math::get_sqrt_price_at_tick(tick_lower)?,
            tick_math::get_sqrt_price_at_tick(tick_upper)?,
            liquidity_delta,
        )?;
    }
    Ok((amount_0, amount_1))
}
