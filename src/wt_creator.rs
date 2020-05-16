//! Creates common wavetables.
//!
//! Supports generating some default waveshapes. In the future, it will also
//! support generating tables from alternative sources (algorithms, lists of
//! harmonics etc.).
//!
//! Currently it supports the wave shapes sine, triangle, saw and square, plus
//! pulse width modulated square waves.

use super::Float;
use super::{Wavetable, WavetableRef};

use log::{info, debug, trace, warn};

use std::sync::Arc;

pub struct WtCreator {
}

impl WtCreator {

    // ------------------
    // Default waveshapes
    // ------------------

    // Insert a sine wave into the given table.
    fn insert_sine(table: &mut [Float], _start_freq: Float, _sample_freq: Float) {
        Wavetable::add_sine_wave(table, 1.0, 1.0);
    }

    // Insert a saw wave into the given table.
    //
    // Adds all odd harmonics, subtracts all even harmonics, with reciprocal
    // amplitude.
    //
    fn insert_saw(table: &mut [Float], start_freq: Float, sample_freq: Float) {
        let num_harmonics = Wavetable::calc_num_harmonics(start_freq * 2.0, sample_freq);
        let mut sign: Float;
        for i in 1..num_harmonics + 1 {
            sign = if (i & 1) == 0 { 1.0 } else { -1.0 };
            Wavetable::add_sine_wave(table, i as Float, 1.0 / i as Float * sign);
        }
        Wavetable::normalize(table);
        // Shift by 180 degrees to keep it symmetrical to Sine wave
        Wavetable::shift(table, table.len() & 0xFFFFFFFC, table.len() / 2);
    }

    // Insert a saw wave into the given table.
    //
    // Adds all harmonics. Should be wrong, but sounds the same.
    //
    fn insert_saw_2(table: &mut [Float], start_freq: Float, sample_freq: Float) {
        let num_harmonics = Wavetable::calc_num_harmonics(start_freq * 2.0, sample_freq);
        for i in 1..num_harmonics + 1 {
            Wavetable::add_sine_wave(table, i as Float, 1.0 / i as Float);
        }
        Wavetable::normalize(table);
    }

    // Insert a triangular wave into the given table.
    //
    // Adds odd cosine harmonics with squared odd reciprocal amplitude.
    //
    fn insert_tri(table: &mut [Float], start_freq: Float, sample_freq: Float) {
        let num_harmonics = Wavetable::calc_num_harmonics(start_freq * 2.0, sample_freq);
        for i in (1..num_harmonics + 1).step_by(2) {
            Wavetable::add_cosine_wave(table, i as Float, 1.0 / ((i * i) as Float));
        }
        Wavetable::normalize(table);
        // Shift by 90 degrees to keep it symmetrical to Sine wave
        Wavetable::shift(table, table.len() & 0xFFFFFFFC, table.len() / 4);
    }

    // Insert a square wave into the given table.
    //
    // Adds odd sine harmonics with odd reciprocal amplitude.
    //
    fn insert_square(table: &mut [Float], start_freq: Float, sample_freq: Float) {
        let num_harmonics = Wavetable::calc_num_harmonics(start_freq * 2.0, sample_freq);
        for i in (1..num_harmonics + 1).step_by(2) {
            Wavetable::add_sine_wave(table, i as Float, 1.0 / i as Float);
        }
        Wavetable::normalize(table);
    }

    /// Create collection of tables with common waveforms.
    ///
    /// The added waveforms are sine, triangle, saw and square. It will generate
    /// tables with 2048 samples, one octave per table, for 11 octaves.
    ///
    /// ```
    /// use wavetable::WtCreator;
    ///
    /// let default_waves = WtCreator::create_default_waves(44100.0);
    /// ```
    pub fn create_default_waves(sample_rate: Float) -> WavetableRef {
        debug!("Initializing default waveshapes");
        let mut wt = Wavetable::new(4, 11, 2048);
        let start_freq = Wavetable::get_start_frequency(440.0);
        wt.insert_tables(0, start_freq, sample_rate, WtCreator::insert_sine);
        wt.insert_tables(1, start_freq, sample_rate, WtCreator::insert_tri);
        wt.insert_tables(2, start_freq, sample_rate, WtCreator::insert_saw);
        wt.insert_tables(3, start_freq, sample_rate, WtCreator::insert_square);
        Arc::new(wt)
    }

    /// Create collection of square waves with different pulse width modulation.
    ///
    /// Creating 64 PWM waves is usually a good compromise between resolution
    /// and memory usage.
    ///
    /// ```
    /// use wavetable::WtCreator;
    ///
    /// let pwm_waves = WtCreator::create_pwm_waves(44100.0, 64);
    /// ```
    pub fn create_pwm_waves(sample_rate: Float, num_pwm_tables: usize) -> WavetableRef {
        info!("Initializing PWM table");

        // Create temporary table containing a saw waveshape
        let num_samples = 2048;
        let mut saw_wt = Wavetable::new(1, 11, num_samples);
        let start_freq = Wavetable::get_start_frequency(440.0);
        saw_wt.insert_tables(0, start_freq, sample_rate, WtCreator::insert_saw);

        let saw_wave = &saw_wt.table[0];
        info!("num_samples: {}", num_samples);
        let mut wt = Wavetable::new(num_pwm_tables, 11, num_samples); // TODO: Get rid of magic number of octaves
        for i in 0..num_pwm_tables {
            // Offset the offset by 1 to keep modulation inside of 100%
            wt.combine_tables(i, &saw_wave, &saw_wave, (i + 1) as Float / (num_pwm_tables + 2) as Float);
        }
        Arc::new(wt)
    }
}
