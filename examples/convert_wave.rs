use wavetable::{WavHandler, WtReader};

use flexi_logger::{Logger, opt_format};

/// Convert a wave file to a different format.
fn main () {
    // Start as "RUST_LOG=info cargo run --example load_wavetable <filename>"
    // to show log info
    Logger::with_env_or_str("myprog=debug, mylib=warn")
                            .format(opt_format)
                            .start()
                            .unwrap();

    let filename = std::env::args().nth(1).expect("Please give name of wave file to load as argument");
    let result = WavHandler::read_file(&filename);
    let wav_file = match result {
        Ok(wav_file) => wav_file,
        Err(_) => return,
    };
    println!("{} bytes with {} bits per sample, resulting in {} samples in {} channels, format {}",
        wav_file.get_data_size(),
        wav_file.get_fmt().get_bits_per_sample(),
        wav_file.get_num_samples(),
        wav_file.get_fmt().get_num_channels(),
        wav_file.get_data().get_type());

    let reader = WtReader::new(".");
    let result = WtReader::create_wavetable(&wav_file, Some(wav_file.get_num_samples()));
    match result {
        Ok(wt_ref) => {
            println!("Got wavetable with {} tables, for {} octaves, with {} samples and {} values per octave",
                wt_ref.num_tables,
                wt_ref.num_octaves,
                wt_ref.num_samples,
                wt_ref.num_values);
            reader.write_file(wt_ref, "out.wav").unwrap();
        },
        Err(_) => {
            println!("Failed to load file {} as wave table", filename);
        }
    }
}
