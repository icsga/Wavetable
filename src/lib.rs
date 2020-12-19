//! A library for creating and using wavetables for sound generation.

#![allow(dead_code)]
#![allow(unused_imports)]

pub mod wavetable;
pub mod wav_handler;
pub mod wt_creator;
pub mod wt_manager;
pub mod wt_oscillator;
pub mod wt_reader;

pub use self::wavetable::{Wavetable, WavetableRef};
pub use wav_handler::{WavHandler, WavData, WavDataType, FmtChunk};
pub use wt_creator::WtCreator;
pub use wt_manager::{WtManager, WtInfo};
pub use wt_oscillator:: WtOsc;
pub use wt_reader::WtReader;

pub type Float = f64;

