use super::Float;
use super::WavetableRef;

use log::{info, trace, warn};

const NUM_SAMPLES_PER_TABLE: usize = 2048;
const NUM_VALUES_PER_TABLE: usize = NUM_SAMPLES_PER_TABLE + 1; // Add one sample for easier interpolation on last sample

pub struct WtOsc {
    pub sample_rate: Float,
    last_update: i64, // Time of last sample generation
    last_sample: Float,
    last_pos: Float,
    wave: WavetableRef,
}

/** Wavetable oscillator implementation.
 *
 * This is a sample implementation of a wavetable oscillator.
 */
impl WtOsc {

    /** Create a new wavetable oscillator.
     *
     * \param sample_rate The global sample rate of the synth
     */
    pub fn new(sample_rate: u32, wave: WavetableRef) -> WtOsc {
        let sample_rate = sample_rate as Float;
        let last_update = 0;
        let last_sample = 0.0;
        let last_pos = 0.0;
        WtOsc{sample_rate,
              last_update,
              last_sample,
              last_pos,
              wave}
    }

    pub fn set_wavetable(&mut self, wavetable: WavetableRef) {
        self.wave = wavetable;
    }

    /** Interpolate between two sample values with the given ratio. */
    fn interpolate(val_a: Float, val_b: Float, ratio: Float) -> Float {
        val_a + ((val_b - val_a) * ratio)
    }

    /** Get a sample from the given table at the given position.
     *
     * Uses linear interpolation for positions that don't map directly to a
     * table index.
     */
    fn get_sample(table: &[Float], table_index: usize, position: Float) -> Float {
        let floor_pos = position as usize;
        let frac = position - floor_pos as Float;
        let position = floor_pos + table_index * NUM_VALUES_PER_TABLE;
        if frac > 0.9 {
            // Close to upper sample
            table[position + 1]
        } else if frac < 0.1 {
            // Close to lower sample
            table[position]
        } else {
            // Interpolate for differences > 10%
            let value_left = table[position];
            let value_right = table[position + 1];
            WtOsc::interpolate(value_left, value_right, frac)
        }
    }

    /* Look up the octave table matching the current frequency. */
    fn get_table_index(num_octaves: usize, freq: Float) -> usize {
        let two: Float = 2.0;
        let mut compare_freq = (440.0 / 32.0) * (two.powf((-9.0) / 12.0));
        for i in 0..num_octaves {
            if freq < compare_freq * 2.0 {
                return i;
            }
            compare_freq *= 2.0;
        }
        0
    }

    pub fn tick(&mut self, frequency: Float, sample_clock: i64, wave_index: Float, reset: bool) -> Float {
        if reset {
            self.reset(sample_clock - 1);
        }

        // Check if we already calculated a matching value
        // TODO: Check if we also need to test the frequency here
        if sample_clock == self.last_update {
            return self.last_sample;
        }

        let dt = sample_clock - self.last_update;
        let dt_f = dt as Float;

        let freq_speed = frequency * (NUM_SAMPLES_PER_TABLE as Float / self.sample_rate);
        let diff = freq_speed * dt_f;
        self.last_pos += diff;
        if self.last_pos > (NUM_SAMPLES_PER_TABLE as Float) {
            // Completed one wave cycle
            self.last_pos -= NUM_SAMPLES_PER_TABLE as Float;
        }

        // TODO: Improve interpolation
        let translated_index = (self.wave.table.len() - 1) as Float * wave_index;
        let lower_wave = translated_index as usize;
        let lower_wave_float = lower_wave as Float;
        let lower_fract: Float = 1.0 - (translated_index - lower_wave_float);
        let upper_fract: Float = if lower_fract != 1.0 { 1.0 - lower_fract } else { 0.0 };

        let table_index = WtOsc::get_table_index(self.wave.num_octaves, frequency);

        let mut result = WtOsc::get_sample(&self.wave.table[lower_wave], table_index, self.last_pos) * lower_fract;
        if upper_fract > 0.0 {
            result += WtOsc::get_sample(&self.wave.table[lower_wave + 1], table_index, self.last_pos) * upper_fract;
        }
        self.last_update += dt;
        self.last_sample = result;
        result
    }

    fn reset(&mut self, sample_clock: i64) {
        self.last_pos = 0.0;
        self.last_update = sample_clock;
    }
}

/*
#[cfg(test)]
#[test]
fn test_calc_num_harmonics() {
    // Base frequency: 2 Hz
    // Sample frequency 20 Hz
    // Nyquist: 10 Hz
    // Num harmonics: [2,] 4, 6, 8 = 3
    assert_eq!(WtOsc::calc_num_harmonics(2.0, 20.0), 3);
}

#[test]
fn test_get_table_index() {
    assert_eq!(WtOsc::get_table_index(10.0), 0);
    assert_eq!(WtOsc::get_table_index(20.0), 1);
    assert_eq!(WtOsc::get_table_index(40.0), 2);
    assert_eq!(WtOsc::get_table_index(80.0), 3);
    assert_eq!(WtOsc::get_table_index(160.0), 4);
    assert_eq!(WtOsc::get_table_index(320.0), 5);
    assert_eq!(WtOsc::get_table_index(640.0), 6);
    assert_eq!(WtOsc::get_table_index(1280.0), 7);
    assert_eq!(WtOsc::get_table_index(2560.0), 8);
    assert_eq!(WtOsc::get_table_index(5120.0), 9);
    assert_eq!(WtOsc::get_table_index(10240.0), 10);
    assert_eq!(WtOsc::get_table_index(20480.0), 10);
}

#[test]
fn test_interpolate() {
    assert_eq!(WtOsc::interpolate(2.0, 3.0, 0.0), 2.0); // Exactly left value
    assert_eq!(WtOsc::interpolate(2.0, 3.0, 1.0), 3.0); // Exactly right value
    assert_eq!(WtOsc::interpolate(2.0, 3.0, 0.5), 2.5); // Middle
}

#[test]
fn test_get_sample() {
    //fn get_sample(table: &[Float], table_index: usize, position: Float) -> Float{
    let mut table = [0.0; NUM_VALUES_PER_TABLE];
    table[0] = 2.0;
    table[1] = 3.0;
    assert_eq!(WtOsc::get_sample(&table, 0, 0.0), 2.0); // Exactly first value
    assert_eq!(WtOsc::get_sample(&table, 0, 1.0), 3.0); // Exactly second value
    assert_eq!(WtOsc::get_sample(&table, 0, 0.5), 2.5); // Middle
    assert_eq!(WtOsc::get_sample(&table, 0, 0.09), 2.0); // Close to first
    assert_eq!(WtOsc::get_sample(&table, 0, 0.99), 3.0); // Close to second
}
*/
