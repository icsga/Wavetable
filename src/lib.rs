#![allow(dead_code)]
#![allow(unused_imports)]

mod wavetable;
mod wt_manager;
mod wt_oscillator;
mod wt_reader;

use wavetable::{Wavetable, WavetableRef};
use wt_manager::WtManager;
use wt_reader::WtReader;

type Float = f64;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
