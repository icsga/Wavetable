extern crate wavetable;

use wavetable::Wavetable;
use wavetable::WtManager;

fn main() {
    let sample_rate = 44100.0;
    let basic_wave_id = 0;

    // Create the wavetable manager and add the default waveshapes to
    // the cache.
    //
    let mut wt_manager = WtManager::new(sample_rate);
    wt_manager.add_basic_tables(basic_wave_id);   // Add the basic waveshapes with ID 0

    // Do an FFT of the default waves.
    // The result will be a vector with a list of the amplitudes of the first
    // 1024 harmonics of the 4 basic waveshapes (Float[4][1024]).
    //
    let num_harmonics = 1024; // 1024 harmonics for a wave of len 2048
    let wt_basic = if let Some(table) = wt_manager.get_table(basic_wave_id) {
        table
    } else {
        panic!();
    };
    let harmonics = wt_basic.convert_to_harmonics(num_harmonics);

    // Generate a new wavetable from the list of harmonics
    //
    let mut wt_new = Wavetable::new(4, 11, 2048); // Reserve space for 4 waveshapes with 11 octave tables each
    wt_new.insert_harmonics(&harmonics, sample_rate).unwrap();

    // Print the four waves to stdout (pipe output to file to plot it with gnoplot)
    //
    let octave = 0; // 0 = lowest. Change to a value from 1 to 10 to show the higher octaves with fewer harmonics
    for i in 0..wt_new.table.len() {    // For all waveshapes
        let table = &wt_new.table[i];
        for k in 0..2048 {              // For all samples of one octave table
            println!("{}: {}", k + (i * 2048), table[k + (octave * 2049)]);
        }
    }
}







