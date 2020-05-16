extern crate wavetable;

use wavetable::WtManager;
use wavetable::WtOsc;
use wavetable::Float;

fn main() {
    let sample_rate = 44100.0;

    // Create the wavetable manager and add the default waveshapes to
    // the cache.
    let mut wt_manager = WtManager::new(sample_rate, "test");
    wt_manager.add_basic_tables(0);   // Add the basic waveshapes as ID 0
    wt_manager.add_pwm_tables(1, 64); // Add 64 PWM waveshapes as ID 1

    // Create a simple wavetable oscillator, using the basic waveshapes.
    let mut osc = WtOsc::new(sample_rate, wt_manager.get_table(0).unwrap());

    // Get some samples of a saw save. We are printing the values of one cycle
    // in 40 steps. This should roughly show values going from +1.0 to -1.0.
    // Due to the bandlimiting, this will not be a straight line if plotted.
    //
    // The get_sample function needs a wave index, which tells it which
    // waveshape to use. There are 4 basic waveshapes in the table, with the
    // index going from 0.0 to 1.0. To get to the third table, we use the index
    // (target table - 1 / number of tables - 1) = (2.0 / 3.0).

    let mut sample: Float;
    let num_samples = 40;
    let table_index = 2.0 / 3.0;
    let freq = (44100.0 / 2048.0) * (2048.0 / num_samples as Float);
    println!("First {} samples of the saw wave at frequency {}:", num_samples, freq);
    for i in 1..num_samples + 1 {
        sample = osc.get_sample(freq, i, table_index, false);
        println!("{}: {}", i, sample);
    }
}
