#![allow(dead_code)]
#![allow(unused_imports)]

mod wavetable;
mod wt_creator;
mod wt_manager;
mod wt_oscillator;
mod wt_reader;

pub use wavetable::{Wavetable, WavetableRef};
use wt_creator::WtCreator;
pub use wt_manager::WtManager;
pub use wt_oscillator:: WtOsc;
use wt_reader::WtReader;

pub type Float = f64;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
