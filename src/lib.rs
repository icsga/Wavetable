//! A library for creating and using wavetables for sound generation.

pub mod wavetable;
pub mod wav_data;
pub mod wav_handler;
pub mod wt_creator;
pub mod wt_manager;
pub mod wt_oscillator;
pub mod wt_reader;

pub use self::wavetable::{Wavetable, WavetableRef, Harmonic};
pub use wav_data::{WavData, WavDataType, Chunk, FmtChunk};
pub use wav_handler::WavHandler;
pub use wt_creator::WtCreator;
pub use wt_manager::{WtManager, WtInfo};
pub use wt_oscillator:: WtOsc;
pub use wt_reader::WtReader;

#[cfg(not(feature = "use_double_precision"))]
pub type Float = f32;

#[cfg(feature = "use_double_precision")]
pub type Float = f64;

