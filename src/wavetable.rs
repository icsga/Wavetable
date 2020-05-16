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
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

use std::cmp;
use std::sync::Arc;

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
    /// assert_eq!(num_harmonics, 1);
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

    /// Convert a wavetable to a vector of harmonics lists.
    ///
    /// Creates one harmonic list for each waveshape in the table. The result
    /// will contain at most <num_harmonics> harmonics in the list.
    ///
    /// ```
    /// use wavetable::Wavetable;
    /// use wavetable::WtManager;
    ///
    /// let mut wt_manager = WtManager::new(44100.0, "data");
    /// wt_manager.add_basic_tables(0);
    /// let wt = if let Some(table) = wt_manager.get_table(0) { table } else { panic!(); };
    /// let harmonics = wt.convert_to_harmonics(1300);
    /// ```
    pub fn convert_to_harmonics(&self, num_harmonics: usize) -> Vec<Vec<Float>> {
        // Allocate memory
        let mut harmonics = vec![vec![0.0; num_harmonics]; self.table.len()];

        // Prepare FFT
        let mut input: Vec<Complex<Float>> = vec![Complex::zero(); 2048];
        let mut output: Vec<Complex<Float>> = vec![Complex::zero(); 2048];
        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(2048);
        let mut value: Float;

        // For all waveshapes in the Wavetable
        for i in 0..self.table.len() {

            // Copy wave to input.
            // We're only checking octave table 1, since it has the most
            // harmonics.
            let table = &self.table[i];
            for j in 0..2048 {
                input[j].re = table[j];
            }

            // Process input
            fft.process(&mut input, &mut output);

            // Find scale of output
            let mut max: Float = 0.0;
            for j in 0..output.len() {
                max = max.max(output[j].im.abs());
            }
            // Transfer output to harmonics list
            for j in 0..num_harmonics {
                value = output[j].im;
                harmonics[i][j] = if value < 0.001 && value > -0.001 {
                    0.0
                } else {
                    (value / max) * -1.0
                };
            }
        }
        harmonics
    }

    // Add a wave with given frequency to a table.
    //
    // Frequency is relative to the buffer length, so a value of 1 will put one
    // wave period into the table. The values are added to the values already
    // in the table. Giving a negative amplitude will subtract the values.
    //
    // The last sample in the table receives the same value as the first, to
    // allow more efficient interpolation (eliminates the need for index
    // wrapping).
    //
    // wave_func is a function receiving an input in the range [0:1] and
    // returning a value in the same range.
    //
    fn add_wave(table: &mut [Float], freq: Float, amplitude: Float, wave_func: fn(Float) -> Float) {
        let extra_sample = table.len() & 0x01;
        let num_samples = table.len() - extra_sample;
        let num_samples_f = num_samples as Float;
        let mult = freq * 2.0 * std::f64::consts::PI;
        let mut position: Float;
        for i in 0..num_samples {
            position = mult * (i as Float / num_samples_f);
            table[i] = table[i] + wave_func(position) * amplitude;
        }
        if extra_sample > 0 {
            table[table.len() - 1] = table[0]; // Add extra sample for easy interpolation
        }
    }

    /// Add a sine wave with given frequency and amplitude to the buffer.
    ///
    /// ```
    /// use wavetable::Wavetable;
    ///
    /// let mut wt = Wavetable::new(4, 11, 2048);
    /// let mut first_wave = wt.get_wave_mut(0);
    /// let frequency = 1.0; // wavelength = buffer size
    /// let amplitude = 1.0;
    /// Wavetable::add_sine_wave(&mut first_wave, frequency, amplitude);
    /// ```
    pub fn add_sine_wave(table: &mut [Float], freq: Float, amplitude: Float) {
        Wavetable::add_wave(table, freq, amplitude, f64::sin);
    }

    /// Add a cosine wave with given frequency and amplitude to the buffer.
    ///
    /// ```
    /// use wavetable::Wavetable;
    ///
    /// let mut wt = Wavetable::new(4, 11, 2048);
    /// let mut first_wave = wt.get_wave_mut(0);
    /// let frequency = 1.0; // wavelength = buffer size
    /// let amplitude = 1.0;
    /// Wavetable::add_cosine_wave(&mut first_wave, frequency, amplitude);
    /// ```
    pub fn add_cosine_wave(table: &mut [Float], freq: Float, amplitude: Float) {
        Wavetable::add_wave(table, freq, amplitude, f64::cos);
    }

    /// Create octave tables with given insert function.
    ///
    /// Divides the given table into NUM_TABLES subtables and uses the given
    /// insert function to insert waveforms into them. Each table serves the
    /// frequency range of one octave.
    ///
    pub fn insert_tables(&mut self,
                         table_id: usize,
                         start_freq: Float,
                         sample_freq: Float,
                         insert_wave: fn(&mut [Float], Float, Float)) {
        info!("Creating table {}", table_id);
        let num_octaves = self.num_octaves;
        let num_values = self.num_values;
        let table = self.get_wave_mut(table_id);
        let mut current_freq = start_freq;
        for i in 0..num_octaves {
            let from = i * num_values;
            let to = (i + 1) * num_values;
            insert_wave(&mut table[from..to], current_freq, sample_freq);
            current_freq *= 2.0; // Next octave
        }
    }

    /// Calculate the start frequency to use for wave creation.
    pub fn get_start_frequency(base_freq: Float) -> Float {
        let two: Float = 2.0;
        (base_freq / 32.0) * (two.powf((-9.0) / 12.0))
    }

    /// Insert a wave from a list of harmonic amplitudes.
    ///
    /// The list of harmonics containes their relative amplitude. After adding
    /// the harmonics to the wavetable, the total amplitude will be normalized.
    ///
    /// ```
    /// use wavetable::Wavetable;
    ///
    /// let harmonics = vec![vec![0.0; 2048]];
    /// let mut wt = Wavetable::new(1, 11, 2048);
    /// wt.insert_harmonics(&harmonics, 44100.0);
    /// ```
    pub fn insert_harmonics(&mut self, harmonics: &[Vec<Float>], sample_freq: Float) -> Result<(), ()> {
        if harmonics.len() != self.table.len() {
            // Number of waveshapes doesn't match
            info!("Wrong number of tables");
            return Err(());
        }

        let mut amp: Float;
        let num_values = self.num_values;

        for i in 0..harmonics.len() { // For each waveshape
            let table = &mut self.table[i];
            let mut start_freq = Wavetable::get_start_frequency(440.0);

            for j in 0..self.num_octaves { // For each octave
                // Calc number of harmonics for this octave
                start_freq *= 2.0;
                let num_harmonics = Wavetable::calc_num_harmonics(start_freq, sample_freq);
                let num_harmonics = cmp::min(num_harmonics, harmonics[i].len());
                let from = j * num_values;
                let to = (j + 1) * num_values;

                // Insert waves
                for freq in 0..num_harmonics {
                    amp = harmonics[i][freq];
                    Wavetable::add_sine_wave(&mut table[from..to], freq as Float, amp);
                }
                Wavetable::normalize(&mut table[from..to]);
            }
        }
        Ok(())
    }

    /// Combine two tables by subtracting one from the other.
    ///
    /// \param table_id ID of the target table to write to
    /// \param table_a Source table that is subtracted from
    /// \param table_b Source table that gets subtracted from table_a
    /// \param offset_b Offset into source table_b (0.0 - 1.0)
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
        info!("Combining source tables into table {}, offset {}", table_id, offset_b);
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
        for i in 0..table.len() {
            current = table[i].abs();
            if current > max {
                max = current;
            }
        }
        for i in 0..table.len() {
            table[i] = table[i] / max;
        }
    }

    /// Shifts all values in a table by the given offset.
    pub fn shift(table: &mut [Float], num_values: usize, offset: usize) {
        let mut temp = vec!(0.0; num_values);
        let mut offset = offset;
        for i in 0..num_values {
            temp[offset] = table[i];
            offset += 1;
            if offset == num_values {
                offset = 0;
            }
        }
        for i in 0..num_values {
            table[i] = temp[i]; // Copy back
        }
        table[num_values] = table[0];
    }

    /// Return min and max values of given table.
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

// TODO: Add tests for wave generation
