use js_sys::{Array,BigInt,Number};

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn hello() -> Number {
    Number::from(23)
}