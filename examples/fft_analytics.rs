extern crate wavetable;

use wavetable::Wavetable;
use wavetable::WtManager;

use std::sync::Arc;

fn main() {
    let sample_rate = 44100.0;

    // Create the wavetable manager and add the default waveshapes to
    // the cache.
    //
    let mut wt_manager = WtManager::new(sample_rate);
    let basic_wave_id = 0;
    wt_manager.add_basic_tables(basic_wave_id);   // Add the basic waveshapes with ID 0

    // Do an FFT of the default waves.
    // The result will be a vector with a list of the amplitudes of the first
    // 1024 harmonics of the 4 basic waveshapes (Float[4][1024], assuming the
    // Wavetable has a length of 2048 samples).
    //
    let wt_basic = if let Some(table) = wt_manager.get_table(basic_wave_id) {
        table
    } else {
        panic!();
    };
    let harmonics = wt_basic.convert_to_harmonics();

    // Generate a new wavetable from the list of harmonics
    //
    let mut wt_new = Wavetable::new(4, 11, 2048); // Reserve space for 4 waveshapes with 11 octave tables each
    wt_new.insert_harmonics(&harmonics, sample_rate).unwrap();
    
    let wt_new = Arc::new(wt_new);
    wt_manager.write_table(wt_new, "out.wav");
}







