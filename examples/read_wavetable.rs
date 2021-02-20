use wavetable::WtReader;

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
    let reader = WtReader::new(".");
    let result = reader.read_file(&filename, Some(2048));
    match result {
        Ok(wt_ref) => {
            println!("Got wavetable with {} tables, for {} octaves, with {} samples and {} values per octave",
                wt_ref.num_tables,
                wt_ref.num_octaves,
                wt_ref.num_samples,
                wt_ref.num_values);
        },
        Err(_) => {
            println!("Failed to load file {} as wave table", filename);
        }
    }
}
