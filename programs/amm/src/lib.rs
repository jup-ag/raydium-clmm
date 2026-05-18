pub mod error;
pub mod instructions;
pub mod libraries;
pub mod states;
pub mod util;

use anchor_lang::prelude::*;
use core as core_;
use instructions::*;
use states::*;

#[cfg(feature = "devnet")]
declare_id!("devi51mZmdwUJGU9hjN27vEz64Gps7uUefqxg27EAtH");
#[cfg(not(feature = "devnet"))]
declare_id!("CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK");
