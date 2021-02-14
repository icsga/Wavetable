use wavetable::WavHandler;

use flexi_logger::{Logger, opt_format};

/// Load a wave file into memory and show some information about the data.
fn main () {
    // Start as "RUST_LOG=info cargo run --example load_wavetable <filename>"
    // to show log info
    Logger::with_env_or_str("myprog=debug, mylib=warn")
                            .format(opt_format)
                            .start()
                            .unwrap();

    let filename = std::env::args().nth(1).expect("Please give name of wave file to analyze as argument");
    let result = WavHandler::read_file(&filename);
    match result {
        Ok(wav_file) => {
            let info = wav_file.get_fmt();
            let data = wav_file.get_data();
            println!("{}:", filename);
            println!("{} bytes with {} bits per sample, resulting in {} samples in {} channels, format {}",
                wav_file.get_data_size(),
                info.get_bits_per_sample(),
                data.get_num_samples(),
                info.get_num_channels(),
                data.get_type());
        },
        Err(()) => println!("Failed to read file {}", filename),
    }
}

