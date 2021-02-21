use wavetable::{WtManager, WtInfo};

use flexi_logger::{Logger, opt_format};

/// Load a wave file as wave table.
fn main () {
    // Start as "RUST_LOG=info cargo run --example load_wavetable <filename>"
    // to show log info
    Logger::with_env_or_str("myprog=debug, mylib=warn")
                            .format(opt_format)
                            .start()
                            .unwrap();

    let filename = std::env::args().nth(1).expect("Please give name of wave file to load as argument");
    let mut wt_manager = WtManager::new(44100.0);
    wt_manager.add_basic_tables(0);
    let fallback = wt_manager.get_table(0).unwrap();
    let mut wt_info = WtInfo{
        id: 1,
        valid: false,
        name: filename.clone(),
        filename: filename.clone()};
    wt_manager.load_table(&mut wt_info, fallback, true);
    let result = wt_manager.get_table(1);
    match result {
        Some(wt_ref) => {
            println!("Got wavetable with {} tables, for {} octaves, with {} samples and {} values per octave",
                wt_ref.num_tables,
                wt_ref.num_octaves,
                wt_ref.num_samples,
                wt_ref.num_values);
            println!("Length of one table: {} samples", wt_ref.table[0].len());

            // Write bandlimited data of single table to file
            wt_manager.write_table(wt_ref, "out.wav");
        },
        None => {
            println!("Failed to load file {} as wave table", filename);
        }
    }
}
