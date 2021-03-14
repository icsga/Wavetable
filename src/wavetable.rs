//! A wavetable representing a collection of waveshapes.
//!
//! A wavetable consists of a collection of waveshapes. Every waveshape in the
//! wavetable itself contains multiple bandlimited tables for use in different
//! octaves. Every octave table is a vector of usually 2048 values
//! representing a single wave cycle.
//!
//! In memory, the table is stored as a vector of vectors. The inner vector
//! holds the samples of a single wave cycle, with the different octave tables
//! stored as a contiguous piece of memory. The outer vector holds the different
//! waveshapes.
//!
//! An oscillator using a wavetable then needs three positions to calculate a
//! sample: The wave index, which determines which waveshape table to use, the
//! current octave, which determines the octave table to use, and the position
//! inside the single wave cycle. Both the wave index and the position are float
//! values, which means some interpolation is required to get the final sample.

use super::Float;

use log::{info, debug, trace, warn};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

use crossbeam;
use scoped_threadpool::Pool;

use std::cmp;
use std::fmt;
use std::sync::Arc;

#[cfg(feature = "use_double_precision")]
pub const PI: f64 = std::f64::consts::PI;
#[cfg(not(feature = "use_double_precision"))]
pub const PI: f32 = std::f32::consts::PI;

#[cfg(feature = "use_double_precision")]
pub const SIN_FUNC: fn(Float) -> Float = f64::sin;
#[cfg(not(feature = "use_double_precision"))]
pub const SIN_FUNC: fn(Float) -> Float = f32::sin;

#[cfg(feature = "use_double_precision")]
pub const COS_FUNC: fn(Float) -> Float = f64::cos;
#[cfg(not(feature = "use_double_precision"))]
pub const COS_FUNC: fn(Float) -> Float = f32::cos;

// Public error types

#[derive(Debug)]
pub struct WrongTableSize;
impl fmt::Display for WrongTableSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Wrong number of tables")
    }
}
impl std::error::Error for WrongTableSize { }

pub type Harmonic = Complex<Float>;

#[derive(Debug)]
pub struct Wavetable {
    pub num_tables: usize,  // Number of different waveshapes
    pub num_octaves: usize, // Number of octave tables to generate per waveshape
    pub num_values: usize,  // Length of a single octave table, including duplicated first element (usually 2049)
    pub num_samples: usize, // Length of a single octave table - 1, actual number of unique values (usually 2048)
    pub table: Vec<Vec<Float>>, // Vector of vectors holding all tables
}

pub type WavetableRef = Arc<Wavetable>;

impl Wavetable {
    /// Creates a new Wavetable instance.
    ///
    /// This function allocates the memory required for storing the given
    /// number of tables.
    ///
    /// ```
    /// use wavetable::Wavetable;
    ///
    /// // Allocate a table for 4 waveshapes, each containing bandlimited
    /// // sub-tables for 11 octaves, with each table holding 2048 samples.
    /// let wt = Wavetable::new(4, 11, 2048);
    /// ```
    pub fn new(num_tables: usize, num_octaves: usize, num_samples: usize) -> Wavetable {
        let num_values = num_samples + 1;
        let table = vec!(vec!(0.0; num_values * num_octaves); num_tables);
        debug!("New Wavetable: {} tables for {} octaves, {} samples",
              num_tables, num_octaves, num_samples);
        Wavetable {
            num_tables,
            num_octaves,
            num_values,
            num_samples,
            table
        }
    }

    /// Create a new Wavetable using the provided sample memory.
    ///
    /// ```
    /// use wavetable::Wavetable;
    ///
    /// let num_tables = 4;
    /// let num_octaves = 11;
    /// let num_samples = 2048;
    /// let num_values = num_samples + 1;
    /// let table = vec!(vec!(0.0; num_values * num_octaves); num_tables);
    /// let wt = Wavetable::new_from_vector(num_tables, num_octaves, num_samples, table);
    /// ```
    pub fn new_from_vector(num_tables: usize, num_octaves: usize, num_samples: usize, table: Vec<Vec<Float>>) -> WavetableRef {
        let num_values = num_samples + 1;
        debug!("New Wavetable: {} tables for {} octaves, {} samples",
              num_tables, num_octaves, num_samples);
        Arc::new(Wavetable {
            num_tables,
            num_octaves,
            num_values,
            num_samples,
            table
        })
    }

    /// Return the waveshape at the given index.
    ///
    /// This will return the vector containing all octave tables for a single
    /// waveshape in the Wavetable.
    ///
    /// ```
    /// use wavetable::Wavetable;
    /// 
    /// let wt = Wavetable::new(4, 11, 2048); // Allocate a Wavetable for 4 waveshapes
    /// let first_wave = wt.get_wave(0);      // Get the vector holding the first waveshape
    /// ```
    pub fn get_wave(&self, wave_id: usize) -> &Vec<Float> {
        &self.table[wave_id]
    }

    /// Return a mutable vector for the selected waveshape.
    ///
    /// This will return a mutable vector containing all octave tables for a
    /// single waveshape in the Wavetable.
    ///
    /// ```
    /// use wavetable::Wavetable;
    ///
    /// let mut wt = Wavetable::new(4, 11, 2048);    // Allocate a Wavetable for 4 waveshapes
    /// let mut second_wave = wt.get_wave_mut(1); // Get the vector holding the second waveshape
    /// ```
    pub fn get_wave_mut(&mut self, wave_id: usize) -> &mut Vec<Float> {
        &mut self.table[wave_id]
    }

    // -----------------------------------------
    // Functions for constructing wavetable data
    // -----------------------------------------

    /// Calculates the number of non-aliasing harmonics for one octave.
    ///
    /// Calculates all the harmonics for the octave starting at base_freq that
    /// do not exceed the Nyquist frequency.
    ///
    /// ```
    /// use wavetable::Wavetable;
    ///
    /// let num_harmonics = Wavetable::calc_num_harmonics(20.0, 100.0);
    /// assert_eq!(num_harmonics, 1); // Base freq 20 Hz, first harmonic 40 Hz, Nyquist at 50 Hz
    /// ```
    pub fn calc_num_harmonics(base_freq: Float, sample_freq: Float) -> usize {
        let nyquist_freq = sample_freq / 2.0;
        if base_freq > nyquist_freq {
            return 0;
        }
        let num_harmonics = (nyquist_freq / base_freq) as usize - 1; // Don't count the base frequency itself
        debug!("Base frequency {}: {} harmonics, highest at {} Hz with sample frequency {}",
            base_freq, num_harmonics, base_freq * (num_harmonics + 1) as Float, sample_freq);
        num_harmonics
    }

    /// Calculate the frequency spectra of the waveshapes in the table.
    ///
    /// Creates one spectrum for each waveshape in the table. Each spectrum is
    /// the result of running an FFT over the samples of the waveshape.
    ///
    /// ```
    /// use wavetable::Wavetable;
    /// use wavetable::WtManager;
    ///
    /// let mut wt_manager = WtManager::new(44100.0);
    /// wt_manager.add_basic_tables(0);
    /// let wt = if let Some(table) = wt_manager.get_table(0) { table } else { panic!(); };
    /// let spectrum = wt.get_freq_spectrum();
    /// ```
    pub fn get_freq_spectrum(&self) -> Vec<Vec<Harmonic>> {
        // Allocate memory
        let num_samples = self.num_samples;
        let fft_len = num_samples;
        let mut spectrum = vec![vec![Harmonic::new(0.0, 0.0); fft_len]; self.table.len()];

        // Prepare FFT
        let mut buffer: Vec<Complex<Float>> = vec![Complex::zero(); fft_len];
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_len);

        // For all waveshapes in the Wavetable
        for (i, ref table) in self.table.iter().enumerate() {

            // Copy wave to buffer.
            // We're only checking octave table 1, since it has the most
            // harmonics.
            for (j, sample) in table.iter().take(num_samples).enumerate() {
                buffer[j].re = *sample;
                buffer[j].im = 0.0;
            }

            // Process buffer
            fft.process(&mut buffer);
            for (j, b) in buffer.iter().enumerate() {
                spectrum[i][j] = *b;
            }
        }
        spectrum
    }

    // Add a wave with given frequency, apmlitude and phase to a table.
    //
    // Frequency is relative to the buffer length, so a value of 1 will put one
    // wave period into the table. The values are added to the values already
    // in the table. Giving a negative amplitude will subtract the values.
    //
    // The last sample in the table receives the same value as the first, to
    // allow more efficient interpolation (eliminates the need for index
    // wrapping).
    //
    // wave_func is a function receiving the position within the waveform as value in the range
    // [0:1] and returning the amplitude for that position as a value in the same range.
    //
    fn add_wave(table: &mut [Float],
                freq: Float,
                amplitude: Float,
                phase: Float,
                wave_func: fn(Float) -> Float) {
        let extra_sample = table.len() & 0x01;
        let num_samples = table.len() - extra_sample;
        let num_samples_f = num_samples as Float;
        let mult = freq * 2.0 * PI;
        let mut position: Float;
        for i in 0..num_samples {
            let mut frac = (i as Float / num_samples_f) + phase;
            while frac > 1.0 {
                frac -= 1.0;
            }
            position = mult * frac;
            table[i] += wave_func(position) * amplitude;
        }
        if extra_sample > 0 {
            table[table.len() - 1] = table[0]; // Add extra sample for easy interpolation
        }
    }

    /// Add a sine wave with given frequency and amplitude to the buffer.
    ///
    /// Frequency is relative to the buffer length, so a value of 1 will put one
    /// wave period into the table. The values are added to the values already
    /// in the table. Giving a negative amplitude will subtract the values.
    ///
    /// ```
    /// use wavetable::Wavetable;
    ///
    /// let mut wt = Wavetable::new(4, 11, 2048);
    /// let mut first_wave = wt.get_wave_mut(0);
    /// let frequency = 1.0; // wavelength = buffer size
    /// let amplitude = 1.0;
    /// let phase = 0.0;
    /// Wavetable::add_sine_wave(&mut first_wave, frequency, amplitude, phase);
    /// ```
    pub fn add_sine_wave(table: &mut [Float], freq: Float, amplitude: Float, phase: Float) {
        Wavetable::add_wave(table, freq, amplitude, phase, SIN_FUNC);
    }

    /// Add a cosine wave with given frequency and amplitude to the buffer.
    ///
    /// Frequency is relative to the buffer length, so a value of 1 will put one
    /// wave period into the table. The values are added to the values already
    /// in the table. Giving a negative amplitude will subtract the values.
    ///
    /// ```
    /// use wavetable::Wavetable;
    ///
    /// let mut wt = Wavetable::new(4, 11, 2048);
    /// let mut first_wave = wt.get_wave_mut(0);
    /// let frequency = 1.0; // wavelength = buffer size
    /// let amplitude = 1.0;
    /// let phase = 0.0;
    /// Wavetable::add_cosine_wave(&mut first_wave, frequency, amplitude, phase);
    /// ```
    pub fn add_cosine_wave(table: &mut [Float], freq: Float, amplitude: Float, phase: Float) {
        Wavetable::add_wave(table, freq, amplitude, phase, COS_FUNC);
    }

    /// Create octave tables with given insert function.
    ///
    /// table_id selects the waveshape to insert into. start_freq chooses the
    /// lowest supported frequency. From that frequency, every octave gets
    /// inserted into it's own sub-table.
    ///
    /// insert_wave is a function that gets called with every octave table as
    /// argument, and is responsible for entering the actual data into the
    /// buffer.
    ///
    /// Examples for the usage of insert_tables can be found in wt_creator.
    ///
    pub fn insert_tables(&mut self,
                         table_id: usize,
                         start_freq: Float,
                         sample_freq: Float,
                         phase: Float,
                         insert_wave: fn(&mut [Float], Float, Float, Float)) {
        info!("Creating table {}", table_id);
        let num_octaves = self.num_octaves;
        let num_values = self.num_values;
        let table = self.get_wave_mut(table_id);
        let mut current_freq = start_freq;
        for i in 0..num_octaves {
            let from = i * num_values;
            let to = (i + 1) * num_values;
            insert_wave(&mut table[from..to], current_freq, sample_freq, phase);
            current_freq *= 2.0; // Next octave
        }
    }

    /// Calculate the start frequency to use for wave creation.
    pub fn get_start_frequency(base_freq: Float) -> Float {
        let two: Float = 2.0;
        (base_freq / 32.0) * (two.powf((-9.0) / 12.0))
    }

    fn run_inv_fft(table: &mut [Float],
                   harmonics: &[Harmonic],
                   num_harmonics: usize,
                   planner: &mut FftPlanner<Float>,
                   buffer: &mut[Complex<Float>]) {
        let num_harmonics = if num_harmonics >= (harmonics.len() / 2) {
            harmonics.len() / 2
        } else {
            num_harmonics
        };
        let num_samples = table.len();
        let fft = planner.plan_fft_inverse(harmonics.len());
        for i in 0..harmonics.len() {
            if i < num_harmonics + 1 || i >= harmonics.len() - (num_harmonics + 1) {
                buffer[i] = harmonics[i];
            } else {
                buffer[i] = Complex::zero(); // Zero out freq components we don't want
            }
        }
        fft.process(buffer);

        for i in 0..num_samples {
            table[i] = buffer[i].re;
        }
    }

    pub fn do_add_freqs(table: &mut[Float], harmonics: &[Harmonic], sample_freq: Float, num_octaves: usize, num_samples: usize, num_values: usize) {
        let mut start_freq = Wavetable::get_start_frequency(440.0);
        let mut planner = FftPlanner::new();
        let mut buffer: Vec<Complex<Float>> = vec![Complex::zero(); harmonics.len()];
        for current_octave in 0..num_octaves {
            let num_harmonics = Wavetable::calc_num_harmonics(start_freq, sample_freq);
            let num_harmonics = cmp::min(num_harmonics, harmonics.len());
            let from = current_octave * num_values;
            let to = from + num_samples;
            Wavetable::run_inv_fft(&mut table[from..to], harmonics, num_harmonics, &mut planner, &mut buffer);
            start_freq *= 2.0;
            if num_values != num_samples {
                table[to] = table[from];
            }
        }
        // Normalize all tables
        for i in 0..num_octaves {
            Wavetable::normalize(&mut table[i * num_values..(i + 1) * num_values]);
        }
    }

    /// Insert frequencies into the wavetable.
    ///
    /// The list of harmonics contains the result of an FFT. After adding
    /// the harmonics to the wavetable, the total amplitude will be normalized.
    ///
    ///
    /// ```
    /// use wavetable::{Wavetable, Harmonic};
    ///
    /// let harmonics = vec![vec![Harmonic::new(0.0, 0.0); 2048]];
    /// let mut wt = Wavetable::new(1, 11, 2048);
    /// wt.add_frequencies(&harmonics, 44100.0);
    /// ```
    pub fn add_frequencies(&mut self, freq_spectrum: &[Vec<Harmonic>], sample_freq: Float) -> Result<(), WrongTableSize> {
        if freq_spectrum.len() != self.table.len()
        || freq_spectrum[0].len() != self.num_samples {
            return Err(WrongTableSize); // Number of waveshapes doesn't match
        }
        let num_samples = self.num_samples;
        let num_values = self.num_values;
        let num_octaves = self.num_octaves;

        // TODO: Make number of threads configurable
        let mut pool = Pool::new(1);
        pool.scoped(|scope| {
            let mut i = 0;
            for table in self.table.chunks_mut(1) { // For each waveshape
                scope.execute(move || {
                    Wavetable::do_add_freqs(&mut table[0], &freq_spectrum[i], sample_freq, num_octaves, num_samples, num_values);
                });
                i += 1;
            }
        });
        /*
        for (i, table) in self.table.iter_mut().enumerate() {
            Wavetable::do_add_freqs(table, &freq_spectrum[i], sample_freq, num_octaves, num_samples, num_values, i);
        }
        */

        Ok(())
    }

    /// Combine two tables by subtracting one from the other.
    ///
    /// * table_id is the ID of the target table to write to
    /// * table_a is the source table that is subtracted from
    /// * table_b is the source table that gets subtracted from table_a
    /// * offset_b is the offset into source table_b (0.0 - 1.0)
    ///
    pub fn combine_tables(&mut self,
                          table_id: usize,
                          table_a: &[Float],
                          table_b: &[Float],
                          offset_b: Float) {
        let num_octaves = self.num_octaves;
        let num_values = self.num_values;
        let num_samples = self.num_samples;
        let table = self.get_wave_mut(table_id);
        let offset_b = (num_samples as Float * offset_b) as usize;
        let mut index_b: usize;
        for i in 0..num_octaves {
            let from = i * num_values;
            let to = (i + 1) * num_values;
            for j in from..to {
                index_b = j + offset_b;
                if index_b >= to {
                    index_b -= num_samples;
                }
                table[j] = table_a[j] - table_b[index_b];
            }
            Wavetable::expand(&mut table[from..to]);
        }
    }

    /// Normalizes samples in a table to the range [-1.0,1.0].
    ///
    /// Searches the maximum absolute value and uses it to calculate the
    /// required scale. Assumes that the values are centered around 0.0.
    ///
    pub fn normalize(table: &mut [Float]) {
        let mut max = 0.0;
        let mut current: Float;
        for sample in table.iter() {
            current = sample.abs();
            if current > max {
                max = current;
            }
        }
        for sample in &mut table.iter_mut() {
            if *sample != 0.0 {
                *sample /= max;
            }
        }
    }

    /// Shifts the position of all values in a table by the given offset.
    pub fn shift(table: &mut [Float], num_values: usize, offset: usize) {
        let mut temp = vec!(0.0; num_values);
        let mut offset = offset;
        for sample in table.iter() {
            temp[offset] = *sample;
            offset += 1;
            if offset == num_values {
                offset = 0;
            }
        }
        table[..num_values].clone_from_slice(&temp[..num_values]);
        table[num_values] = table[0];
    }

    // Return min and max values of given table.
    fn get_extremes(table: &[Float]) -> (Float, Float) {
        let mut max = -1.0;
        let mut min = 1.0;
        let mut current: Float;
        for i in 0..table.len() {
            current = table[i];
            if current > max {
                max = current;
            } else if current < min {
                min = current;
            }
        }
        (min, max)
    }

    /// Expand the samples in a table to the rage [-1.0, 1.0].
    ///
    /// Scales and shifts a wave to fit into the target range. Uses the minimum
    /// and the maximum of the values to calculate scale factor and offset.
    ///
    pub fn expand(table: &mut [Float]) {
        let (min, max) = Wavetable::get_extremes(table);
        let scale = 2.0 / (max - min);
        let offset = (max + min) / 2.0;
        let mut new_val: Float;
        for i in 0..table.len() {
            new_val = (table[i] - offset) * scale;
            table[i] = new_val;
        }
    }

    /// Print all values of a table to the logfile.
    pub fn show(&self) {
        for t in &self.table {
            for (i, s) in t.iter().enumerate() {
                info!("{}: {}", i, s);
            }
        }
    }
}

// ----------------------------------------------
//                  Unit tests
// ----------------------------------------------

struct TestContext {
}

impl TestContext {
    pub fn new() -> Self {
        TestContext{}
    }
}

fn is_close_to(actual: Float, expected: Float, delta: Float, index: usize) -> bool {
    let diff = actual - expected;
    if diff > delta || diff < -delta {
        println!("{}: Expected {}, actual {}, delta {}", index, expected, actual, delta);
        false
    } else {
        true
    }
}

fn is_close_to_wrap(actual: Float, expected: Float, delta: Float, wrap: Float, index: usize) -> bool {
    let (mut bigger, smaller) = if actual > expected { (actual, expected) } else { (expected, actual) };
    let mut diff = bigger - wrap;
    while diff > -delta {
        println!("{} close to wrap {}, reducing", bigger, wrap);
        bigger -= wrap;
        diff = bigger - wrap;
    }
    let diff = bigger - smaller;
    if diff < delta {
        return true;
    }
    println!("{}: Expected {}, actual {}, delta {}", index, expected, actual, delta);
    false
}

#[test]
fn single_frequency_can_be_added() {
    let num_samples = 2048;
    let num_tables = 1;
    let num_octaves = 11;
    let freq = 1.0;
    let amp = 1.0;
    let phase = 0.0;

    for i in 0..num_tables {
        let mut wt = Wavetable::new(num_tables, num_octaves, num_samples); // 256 tables, bandlimited for 11 octaves, with 2048 samples each
        let num_values = wt.num_values;
        Wavetable::add_sine_wave(&mut wt.get_wave_mut(i)[0..num_values], freq, amp, phase);

        for j in 0..num_tables {
            let t = &wt.get_wave(j);
            if j != i {
                for x in t.iter() {
                    assert!(*x == 0.0);
                }
            }
        }
    }
}

#[test]
fn single_frequency_can_be_bandlimted() {
    let sample_freq = 44100.0;
    let num_samples = 2048;
    let num_tables = 1;
    let num_octaves = 11;
    let freq = 1.0;
    let amp = 1.0;
    let phase = 0.0;
    let mut wt_ref = Wavetable::new(1, 1, num_samples);
    Wavetable::add_sine_wave(wt_ref.get_wave_mut(0), freq, amp, phase);
    let wave_ref = wt_ref.get_wave(0);
    for i in 0..num_tables {
        println!("Testing position {}", i);
        let mut wt = Wavetable::new(num_tables, num_octaves, num_samples); // 256 tables, bandlimited for 11 octaves, with 2048 samples each
        let num_values = wt.num_values;
        Wavetable::add_sine_wave(&mut wt.get_wave_mut(i)[0..num_values], freq, amp, phase);
        let harmonics = wt.get_freq_spectrum();

        // Assert that only one table gets signal
        for j in 0..num_tables {
            if j == i {
                continue;
            }
            println!("Testing harmonics for table {}", j);
            for h in &harmonics[j] {
                assert!(is_close_to(h.re, 0.0, 0.0000000001, j));
                assert!(is_close_to(h.im, 0.0, 0.0000000001, j));
            }
        }

        let mut wt_new = Wavetable::new(num_tables, num_octaves, num_samples); // 256 tables, bandlimited for 11 octaves, with 2048 samples each
        wt_new.add_frequencies(&harmonics, sample_freq).unwrap();
        for j in 0..num_tables {
            println!("Comparing wave {}", j);
            let t = &wt_new.get_wave(j);
            if j == i {
                for (k, s) in wave_ref.iter().enumerate() {
                    assert!(is_close_to(t[k], *s, 0.00001, k));
                }
            } else {
                for (k, x) in t.iter().enumerate() {
                    assert!(is_close_to(*x, 0.0, 0.00001, k));
                }
            }
        }
    }
}
