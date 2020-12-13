use wavetable::WavHandler;

use flexi_logger::{Logger, opt_format};

fn main () {
    Logger::with_env_or_str("myprog=debug, mylib=warn")
                            .log_to_file()
                            .directory("log_files")
                            .format(opt_format)
                            .start()
                            .unwrap();

    let filename = std::env::args().nth(1).expect("Please give filename to load");

    // Open the file (relative to current path)
    let result = WavHandler::read_file(&filename);
    match result {
        Ok(_wf_ref) => println!("Success"),
        Err(()) => println!("Failed"),
    }
}

