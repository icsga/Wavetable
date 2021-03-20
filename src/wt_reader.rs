//! Reads wavetable files in WAV format.
//!
//! Supports generating the following types of wavetables:
//! 1. Single cycle wave as read from file: This can be an arbitrary length
//!    table which is not interpolated
//! 2. Multiple waves as read from file: This will create multiple arrays of
//!    samples, each containing one cycle of a waveform.
//!
use super::{Wavetable, WavetableRef, WavHandler, WavData, WavDataType};
use super::Float;

use log::{error, info};

use num::ToPrimitive;
use std::mem;
use std::sync::Arc;

//const YAZZ_WT_CHUNK_ID: u32 = 0x7A7A6179;

pub struct WtReader {
    base_path: String,
}

impl WtReader {
    /// Creates a new WtReader instance.
    ///
    /// The argument is a path to a directory that files will be read from.
    ///
    /// ```
    /// use wavetable::WtReader;
    ///
    /// let data_dir = "data".to_string();
    ///
    /// let reader = WtReader::new(&data_dir);
    /// ```
    pub fn new(path: &str) -> Self {
        let mut reader = WtReader{base_path: "".to_string()};
        reader.set_path(path);
        reader
    }

    /// Set the working directory to read files from
    ///
    /// The argument is a path to a directory that files will be read from.
    ///
    /// ```
    /// use wavetable::WtReader;
    ///
    /// let mut reader = WtReader::new("");
    ///
    /// let data_dir = "data";
    /// reader.set_path(&data_dir);
    /// ```
    pub fn set_path(&mut self, path: &str) {
        self.base_path = path.to_string();
        let path_bytes = self.base_path.as_bytes();
        if path_bytes.len() > 0 && path_bytes[path_bytes.len() - 1] != b'/' {
            self.base_path.push('/');
        }
        info!("Set base path to [{}]", self.base_path);
    }

    /// Read a file with the given filename.
    ///
    /// The filename should not contain the full path of the file, but be
    /// relative to the base path set in the constructor.
    ///
    /// The number of samples per table is either passed to the function or
    /// taken from the file itself. If the number is passed in, the file must
    /// contain that exact number or a multiple of it of samples. Each set of
    /// samples is interpreted as a single waveform, creating a separate entry
    /// in the resulting wavetable.
    ///
    /// If the number of samples is not passed in, the whole file is seen as a
    /// single cycle waveform and a single table entry is created.
    ///
    /// ``` no_run
    /// use wavetable::WtReader;
    ///
    /// # fn main() -> Result<(), ()> {
    ///
    /// let reader = WtReader::new("data");
    /// let wavetable = reader.read_file("test", None)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_file(&self, filename: &str, samples_per_table: Option<usize>) -> Result<WavetableRef, ()> {
        if filename == "" {
            return Err(());
        }
        let filename = self.base_path.clone() + filename;
        info!("Reading file [{}]", filename);
        let wav_file = WavHandler::read_file(&filename)?;
        WtReader::create_wavetable(&wav_file, samples_per_table)
    }

    /// Convert a given wave file data to a wavetable.
    ///
    /// Source is wave struct containing sample data in Vec<u8> format.
    /// samples_per_table is the length of a single wave cycle, usually 2048.
    pub fn create_wavetable(wav_file: &WavData, samples_per_table: Option<usize>) -> Result<WavetableRef, ()> {
        if wav_file.get_fmt().num_channels > 1 {
            error!("Only single channel files supported");
            return Err(());
        }
        let table_size = if let Some(num_samples) = samples_per_table { num_samples } else { wav_file.get_samples().get_num_samples() };
        info!("Creating wavetable with {} samples", table_size);
        match &**wav_file.get_samples() {
            WavDataType::PCM8(data) => WtReader::convert_wav_to_table(&data, 0_u8, 255_u8, table_size),
            WavDataType::PCM16(data) => WtReader::convert_wav_to_table(&data, -32768_i16, 32767_i16, table_size),
            WavDataType::FLOAT32(data) => WtReader::convert_wav_to_table(&data, -1.0_f32, 1.0_f32, table_size),
            WavDataType::FLOAT64(data) => WtReader::convert_wav_to_table(&data, -1.0_f64, 1.0_f64, table_size),
        }
    }

    fn convert_wav_to_table<T>(data: &Vec<T>, min: T, max: T, samples_per_table: usize) -> Result<WavetableRef, ()>
            where T: ToPrimitive + Copy + Clone {
        let needs_conversion: bool = (mem::size_of::<T>() != mem::size_of::<Float>())
                                    || data.len() % samples_per_table != 0;
        if needs_conversion {
            WtReader::do_interpolating_conversion(data, min, max, samples_per_table)
        } else {
            WtReader::do_direct_conversion(data, samples_per_table)
        }
    }

    // Create a table and fill it with values by interpolating between neighboring samples, converting to target data type (float).
    fn do_interpolating_conversion<T>(data: &Vec<T>, min: T, max: T, samples_per_table: usize) -> Result<WavetableRef, ()>
            where T: ToPrimitive + Copy + Clone {
        info!("Doing interpolating conversion");
        // Determine wavetable properties
        let num_samples = data.len();
        let mut source_inc: Float = 1.0;
        let num_tables =
            if num_samples < samples_per_table {
                // Single table, interpolate source samples to match table size
                source_inc = (num_samples - 1) as Float / (samples_per_table - 1) as Float;
                1
            } else if num_samples % samples_per_table != 0 {
                // Source isn't multiple of table size, can't handle this case
                info!("Unexpected number of samples: {}, not multiple of {}", num_samples, samples_per_table);
                return Err(());
            } else {
                // One or more tables with exact number of samples
                num_samples / samples_per_table
            };
        info!("{} samples total, {} tables with {} values each", num_samples, num_tables, samples_per_table);

        // Calculate required scaling
        #[cfg(feature = "use_double_precision")]
            let min_f: Float = min.to_f64().unwrap();
        #[cfg(feature = "use_double_precision")]
            let max_f: Float = max.to_f64().unwrap();
        #[cfg(not(feature = "use_double_precision"))]
            let min_f: Float = min.to_f32().unwrap();
        #[cfg(not(feature = "use_double_precision"))]
            let max_f: Float = max.to_f32().unwrap();

        let scale = 2.0 / (max_f - min_f);
        let offset = min_f * scale + 1.0;

        // Convert values
        let mut index_f: Float;
        let mut lower_index: usize;
        let mut upper_index: usize;
        let mut wt = Wavetable::new(num_tables, 1, samples_per_table);
        let mut ip_val: Float;
        let mut fract: Float;
        for i in 0..num_tables {
            let table = &mut wt.table[i];
            for j in 0..samples_per_table {
                index_f = j as Float * source_inc;
                lower_index = index_f as usize;
                upper_index = if lower_index == num_samples - 1 { lower_index } else { lower_index + 1 };
                fract = index_f - ((index_f as usize) as Float);
                match WtReader::interpolate(data[lower_index], data[upper_index], fract) {
                    Some(v) => {
                        ip_val = (v * scale) - offset;
                        table[j] = ip_val as Float;
                    }
                    None => {
                        error!("Failed to convert source samples to target type");
                        return Err(());
                    }
                }
            }
            table[samples_per_table] = table[0]; // Duplicate first entry as last entry for easy interpolation
            Wavetable::normalize(table);
        }
        Ok(Arc::new(wt))
    }

    fn interpolate<S: ToPrimitive + Copy + Clone>(source_lower: S, source_upper: S, fract: Float) -> Option<Float> {
        #[cfg(feature = "use_double_precision")]
            let lower_f: Float = source_lower.to_f64().unwrap();
        #[cfg(feature = "use_double_precision")]
            let upper_f: Float = source_upper.to_f64().unwrap();
        #[cfg(not(feature = "use_double_precision"))]
            let lower_f: Float = source_lower.to_f32().unwrap();
        #[cfg(not(feature = "use_double_precision"))]
            let upper_f: Float = source_upper.to_f32().unwrap();

        Some((lower_f + ((upper_f - lower_f) * fract)) as Float)
    }

    // Create a table and fill it with values directly converted to target data type (float).
    fn do_direct_conversion<T>(data: &Vec<T>, samples_per_table: usize) -> Result<WavetableRef, ()>
            where T: ToPrimitive + Copy + Clone {
        info!("Doing direct conversion");
        // Determine wavetable properties
        let num_samples = data.len();
        let num_tables = num_samples / samples_per_table;

        info!("{} samples total, {} tables with {} values each", num_samples, num_tables, samples_per_table);

        // Convert values
        let mut wt = Wavetable::new(num_tables, 1, samples_per_table);
        for i in 0..num_tables {
            let table = &mut wt.table[i];
            for j in 0..samples_per_table {
                #[cfg(feature = "use_double_precision")] {
                    table[j] = data[(i * samples_per_table) + j].to_f64().unwrap();
                }
                #[cfg(not(feature = "use_double_precision"))] {
                    table[j] = data[(i * samples_per_table) + j].to_f32().unwrap();
                }
            }
            table[samples_per_table] = table[0]; // Duplicate first entry as last entry for easy interpolation
            Wavetable::normalize(table);
        }
        Ok(Arc::new(wt))
    }

    pub fn write_file(&self, wt_ref: WavetableRef, filename: &str) -> Result<(), ()> {
        // TODO: BROKEN! This includes all the double samples at the end of
        // each waveform. We need to split it up into slices to write.
        #[cfg(feature = "use_double_precision")]
            let samples: Box<WavDataType> = Box::new(WavDataType::FLOAT64(wt_ref.table[0].clone()));
        #[cfg(not(feature = "use_double_precision"))]
            let samples: Box<WavDataType> = Box::new(WavDataType::FLOAT32(wt_ref.table[0].clone()));
        let mut wav_data = WavData::new_from_data(samples);
        println!("Added first table, {} samples with size {}",
            wav_data.get_num_samples(), wav_data.get_num_bytes());
        for table in wt_ref.table.iter().skip(1) {
            #[cfg(feature = "use_double_precision")]
                let samples: Box<WavDataType> = Box::new(WavDataType::FLOAT64(table.clone()));
            #[cfg(not(feature = "use_double_precision"))]
                let samples: Box<WavDataType> = Box::new(WavDataType::FLOAT32(table.clone()));
            wav_data.append_samples(samples).unwrap();
        }

        // TODO: Add Wavetable-Lib-specific chunk with WT information
        //       (Samples per table, number of tables)
        //let yazz_chunk = Chunk::new(YAZZ_WT_CHUNK_ID, (mem::size_of::<usize>() * 3) as u32);
        //wav_data.add_chunk(yazz_chunk);

        WavHandler::write_file(&wav_data, filename)
    }
}


// ----------------------------------------------
//                  Unit tests
// ----------------------------------------------

#[cfg(test)]
fn values_match(actual: &Vec<Float>, expected: &Vec<Float>, delta: Float) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    for i in 0..actual.len() {
        let diff = actual[i] - expected[i];
        if diff > delta || diff < -delta {
            println!("Missmatch, actual {}, expected {}", actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

#[test]
fn base_path_is_set_up_correctly() {
    let wtr = WtReader::new("NoSlash");
    assert!(wtr.base_path == "NoSlash/".to_string());

    let wtr = WtReader::new("WithSlash/");
    assert!(wtr.base_path == "WithSlash/".to_string());
}

#[test]
fn u8_can_be_converted() {
    let data = vec![0_u8, 255_u8];
    let wav_data = WavData::new_from_data(Box::new(WavDataType::PCM8(data)));
    let wt = WtReader::create_wavetable(&wav_data, None).unwrap();
    assert!(wt.num_tables == 1);
    assert!(wt.num_octaves == 1);
    assert!(wt.num_samples == 2);
    assert!(wt.num_values == 3);
    assert!(wt.table == vec![vec![-1.0, 1.0, -1.0]]);
}

#[test]
fn i16_can_be_converted() {
    let data = vec![-32768_i16, 32767_i16];
    let wav_data = WavData::new_from_data(Box::new(WavDataType::PCM16(data)));
    let wt = WtReader::create_wavetable(&wav_data, None).unwrap();
    assert!(wt.num_tables == 1);
    assert!(wt.num_octaves == 1);
    assert!(wt.num_samples == 2);
    assert!(wt.num_values == 3);
    assert!(wt.table == vec![vec![-1.0, 1.0, -1.0]]);
}

#[test]
fn f32_can_be_converted() {
    let data = vec![-1.0_f32, -0.1234_f32, 0.0_f32, 0.1234_f32, 1.0_f32];
    let wav_data = WavData::new_from_data(Box::new(WavDataType::FLOAT32(data)));
    let wt = WtReader::create_wavetable(&wav_data, None).unwrap();
    assert!(wt.num_tables == 1);
    assert!(wt.num_octaves == 1);
    assert!(wt.num_samples == 5);
    assert!(wt.num_values == 6);
    assert!(values_match(&wt.table[0], &vec![-1.0, -0.1234, 0.0, 0.1234, 1.0, -1.0], 0.000001));
}

#[test]
fn f64_can_be_converted() {
    let data = vec![-1.0_f64, -0.1234_f64, 0.0_f64, 0.1234_f64, 1.0_f64];
    let wav_data = WavData::new_from_data(Box::new(WavDataType::FLOAT64(data)));
    let wt = WtReader::create_wavetable(&wav_data, None).unwrap();
    assert!(wt.num_tables == 1);
    assert!(wt.num_octaves == 1);
    assert!(wt.num_samples == 5);
    assert!(wt.num_values == 6);
    assert!(values_match(&wt.table[0], &vec![-1.0, -0.1234, 0.0, 0.1234, 1.0, -1.0], 0.000001));
}

#[test]
fn u8_is_interpolated_correctly() {
    let data = vec![0_u8, 64_u8, 128_u8, 192_u8, 255_u8];
    let wav_data = WavData::new_from_data(Box::new(WavDataType::PCM8(data)));
    let wt = WtReader::create_wavetable(&wav_data, None).unwrap();
    assert!(wt.num_tables == 1);
    assert!(wt.num_octaves == 1);
    assert!(wt.num_samples == 5);
    assert!(wt.num_values == 6);
    assert!(values_match(&wt.table[0], &vec![-1.0, -0.5, 0.0, 0.5, 1.0, -1.0], 0.01));
}

#[test]
fn wrong_number_of_channels_is_rejected() {
    let data = vec![0_u8, 64_u8, 128_u8, 192_u8, 255_u8];
    let mut wav_data = WavData::new_from_data(Box::new(WavDataType::PCM8(data)));
    wav_data.get_fmt_mut().num_channels = 2;
    let result = WtReader::create_wavetable(&wav_data, None);
    assert!(if let Err(_) = result { true } else { false });
}

#[test]
fn wrong_number_of_samples_is_rejected() {
    let data = vec![0_u8, 64_u8, 128_u8, 192_u8, 255_u8];
    let wav_data = WavData::new_from_data(Box::new(WavDataType::PCM8(data)));
    let result = WtReader::create_wavetable(&wav_data, Some(4));
    assert!(if let Err(_) = result { true } else { false });
}

#[test]
fn multiple_tables_can_be_read() {
    let data = vec![0_u8, 64_u8, 128_u8, 192_u8, 255_u8, 0_u8, 64_u8, 128_u8, 192_u8, 255_u8];
    let wav_data = WavData::new_from_data(Box::new(WavDataType::PCM8(data)));
    let wt = WtReader::create_wavetable(&wav_data, Some(5)).unwrap();
    assert!(wt.num_tables == 2);
    assert!(wt.num_octaves == 1);
    assert!(wt.num_samples == 5);
    assert!(wt.num_values == 6);
    assert!(values_match(&wt.table[0], &vec![-1.0, -0.5, 0.0, 0.5, 1.0, -1.0], 0.01));
    assert!(values_match(&wt.table[1], &vec![-1.0, -0.5, 0.0, 0.5, 1.0, -1.0], 0.01));
}

#[test]
fn samples_are_interpolated_correctly() {
    let data = vec![-1.0_f32, 1.0_f32];
    let wav_data = WavData::new_from_data(Box::new(WavDataType::FLOAT32(data)));
    let wt = WtReader::create_wavetable(&wav_data, Some(5)).unwrap();
    assert!(wt.num_tables == 1);
    assert!(wt.num_octaves == 1);
    assert!(wt.num_samples == 5);
    assert!(wt.num_values == 6);
    assert!(values_match(&wt.table[0], &vec![-1.0, -0.5, 0.0, 0.5, 1.0, -1.0], 0.000001));
}
