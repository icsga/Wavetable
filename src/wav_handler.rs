use crate::wav_data::*;

use std::fs::File;
use std::io::prelude::*;
use std::io::{Read, Write, BufReader, SeekFrom};
use std::mem;
use std::process::{Command, Stdio};

use log::{debug, error, info, trace};

// List of Chunk IDs as u32 values (little endian)
// TODO: Check how to improve this for portability
const CID_RIFF: u32 = 0x46464952;
const CID_WAVE: u32 = 0x45564157;
const CID_FMT:  u32 = 0x20746d66;
const CID_DATA: u32 = 0x61746164;

const SIZE_WAVE_HEADER: u32 = 4;
const SIZE_CHUNK_HEADER: u32 = 8;
const SIZE_FMT_CHUNK: u32 = SIZE_CHUNK_HEADER + 16;

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
struct ChunkHeader {
    chunk_id: u32,
    size: u32
}

impl ChunkHeader {
    unsafe fn get_size(&self) -> usize {
        self.size as usize
    }
}

pub struct WavHandler;

/// Handles reading and writing of .wav files.
///
/// Reads wave files into memory as vectors of samples. The resulting struct
/// contains the FMT info, the data, and a list of additional chunks that
/// were found in the file (TBD).
///
/// Writes samples in the format provided, including any extra chunks that
/// the WavData object contains.
impl WavHandler {
    /// Read a file with the given filename.
    ///
    ///
    /// ``` no_run
    /// use wavetable::WavHandler;
    ///
    /// # fn main() -> Result<(), ()> {
    ///
    /// let wave_data = WavHandler::read_file("test")?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_file(filename: &str) -> Result<WavData, ()> {
        if filename == "" {
            return Err(());
        }
        info!("Reading wave file [{}]", filename);
        let result = File::open(filename);
        if let Ok(file) = result {
            let reader = BufReader::new(file);
            WavHandler::read_content(reader)
        } else {
            error!("Unable to open file [{}]", filename);
            Err(())
        }
    }

    /// Read wave data from the provided input stream.
    ///
    /// Source is any stream object implementing the Read trait.
    ///
    /// ``` no_run
    /// use wavetable::WavHandler;
    /// use std::io::Cursor;
    ///
    /// # fn main() -> Result<(), ()> {
    ///
    /// let data: &[u8] = &[0x00]; // Some buffer with wave data
    /// let buffer = Cursor::new(data);
    /// let wave_data = WavHandler::read_content(buffer)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_content<R: Read + Seek>(mut source: R) -> Result<WavData, ()> {
        // Read RIFF header and filetype
        let result = WavHandler::read_riff_container(&mut source, CID_WAVE);
        let size = match result {
            Ok(s) => s,
            Err(()) => return Err(()),
        };

        let mut file = WavData::new();
        let mut bytes_read: usize = 4; // Already read the 4 bytes of file type
        let mut fmt_found = false;
        let mut data_found = false;

        // Read chunks
        loop {
            let result: Result<ChunkHeader, ()> = WavHandler::read_object(&mut source);
            if let Ok(header) = result {
                debug!("Reading {} chunk, size {}", WavHandler::get_id_name(header.chunk_id), unsafe{header.get_size()});
                match header.chunk_id {
                    CID_FMT  => {
                        // Get data format
                        WavHandler::read_chunk_into(file.get_fmt_mut(), &mut source, header.size as usize)?;
                        fmt_found = true;
                    }
                    CID_DATA => {
                        // Found data section, read samples
                        let info = file.get_fmt();
                        let fmt_tag = info.format_tag;
                        let bps = info.bits_per_sample;
                        file.set_samples(WavHandler::read_samples(&mut source,
                                                             header.size as usize,
                                                             fmt_tag,
                                                             bps)?
                        );
                        data_found = true;
                    },
                    _ => WavHandler::skip_chunk(&mut source, header.size),
                }
                bytes_read += (header.size + 8) as usize;
            } else {
                break; // Error or finished reading file. In both cases return data read so far.
            }
        }
        if bytes_read == size {
            debug!("Finished reading {} bytes", bytes_read);
        } else {
            error!("Invalid file size, read {} bytes, expected {}", bytes_read, size);
        }
        if !fmt_found {
            error!("Invalid file format, format chunk missing");
            return Err(());
        }
        if !data_found {
            error!("Invalid file format, data chunk missing");
            return Err(());
        }
        Ok(file)
    }

    // Read the RIFF container information from the input stream.
    //
    // This expects a RIFF header, followed by a 4-byte identifier (e.g.
    // "WAVE"), which is passed as argument.
    fn read_riff_container<R: Read>(source: &mut R, expected_cid: u32) -> Result<usize, ()> {
        // Read RIFF header
        let result: Result<ChunkHeader, ()> = WavHandler::read_object(source);
        let data_size = match result {
            Ok(header) => {
                debug!("Reading {} chunk, size {}", WavHandler::get_id_name(header.chunk_id), unsafe{header.get_size()});
                if header.chunk_id != CID_RIFF {
                    error!("Unexpected chunk ID, expected RIFF, found {}", WavHandler::get_id_name(header.chunk_id));
                    return Err(());
                }
                header.size as usize
            }
            Err(()) => return Err(()),
        };
        let result: Result<u32, ()> = WavHandler::read_object(source);
        match result {
            Ok(header) => {
                // RIFF header is followed by 4 bytes giving the file type
                debug!("File type: {}", WavHandler::get_id_name(header));
                if header != expected_cid {
                    error!("Unexpected file type, expected {}, found {}",
                        WavHandler::get_id_name(expected_cid), WavHandler::get_id_name(header));
                    return Err(());
                }
            }
            Err(()) => return Err(()),
        }
        Ok(data_size)
    }

    // Read an arbitrary object from the input stream.
    fn read_object<R: Read, T>(source: &mut R) -> Result<T, ()> {
        let mut object: T = unsafe { mem::zeroed() };
        let object_size = mem::size_of::<T>();
        unsafe {
            let object_slice = std::slice::from_raw_parts_mut(&mut object as *mut _ as *mut u8, object_size);
            if let Err(_) = source.read_exact(object_slice) {
                // If we can't read the full object, we might have reached the
                // end of the file. Return Err to signal nothing was read.
                return Err(());
            }
            trace!("Read object of {} bytes", object_size);
        }
        Ok(object)
    }

    // Read the contents of a chunk from the input stream.
    //
    // The chunk header is assumed to have been read already. This is required
    // for the FMT chunk, since it can have different sizes.
    fn read_chunk_into<R: Read, T: std::fmt::Debug>(target: &mut T, source: &mut R, size: usize) -> Result<(), ()> {
        unsafe {
            let data_slize = std::slice::from_raw_parts_mut(target as *mut _ as *mut u8, size);
            if let Err(_) = source.read_exact(data_slize) {
                error!("Reading chunk data failed");
                return Err(());
            }
            debug!("Read chunk: {:#?}", target);
        }
        Ok(())
    }

    // Read samples into buffer.
    fn read_samples<R: Read>(source: &mut R, num_bytes: usize, format_tag: u16, bits_per_sample: u16) -> Result<DataChunk, ()> {
        let samples = match format_tag {
            FMT_PCM => match bits_per_sample {
                8  => WavHandler::read_samples_into(source, num_bytes, 0 as u8),
                16 => WavHandler::read_samples_into(source, num_bytes, 0 as i16),
                _  => return Err(()),
            },
            FMT_FLOAT => match bits_per_sample {
                32 => WavHandler::read_samples_into(source, num_bytes, 0.0 as f32),
                64 => WavHandler::read_samples_into(source, num_bytes, 0.0 as f64),
                _  => return Err(()),
            },
            _ => return Err(()),
        };
        match samples {
            Ok(s) => Ok(DataChunk::new(num_bytes, Box::new(s))),
            Err(_) => Err(())
        }
    }

    // Read bytes from file and convert to matching data type of samples
    fn read_samples_into<R: Read, T>(source: &mut R, num_bytes: usize, init_val: T) -> Result<WavDataType, ()>
            where T: Copy + Clone + WavDataTypeGetter<T> {
        let mut buf: T = unsafe { mem::zeroed() };
        let sample_size = mem::size_of::<T>();
        let num_samples = num_bytes / sample_size;
        info!("{} samples of size {} bytes", num_samples, sample_size);
        let mut samples: Vec<T> = vec!{init_val; num_samples};
        unsafe {
            let sample = std::slice::from_raw_parts_mut(&mut buf as *mut _ as *mut u8, sample_size);
            for i in 0..num_samples {
                source.read_exact(sample).unwrap();
                samples[i] = buf as T;
            }
        }
        Ok(init_val.get_wav_data_type(samples))
    }

    // Skip over the rest of the current chunk to the next header.
    fn skip_chunk<R: Read + Seek>(source: &mut R, num_bytes: u32) {
        source.seek(SeekFrom::Current(num_bytes as i64)).unwrap();
    }

    // Convert a given chunk ID from u32 to printable string.
    fn get_id_name(value: u32) -> String {
        let bytes = value.to_le_bytes();
        String::from_utf8(bytes.to_vec()).expect("Found invalid UTF-8")
    }

    // ====================
    // Writing of WAV files
    // ====================

    /// Write the given WavData to a file.
    ///
    /// This writes the fmt chunk, any additional chunks and the sample data to
    /// the file.
    pub fn write_file(data: &WavData, filename: &str) -> Result<(), ()> {
        let result = File::create(filename);
        if let Ok(mut file) = result {
            //let writer = BufWriter::new(file);
            WavHandler::write_content(&mut file, data).unwrap();
            Ok(())
        } else {
            error!("Unable to open file [{}]", filename);
            Err(())
        }
    }

    // Write the WAV data to the given output stream.
    fn write_content<W: Write>(dest: &mut W, data: &WavData) -> Result<(), std::io::Error> {
        // Calculate size:
        // - 4 bytes for "WAVE" header
        // - 20 bytes for fmt chunk
        // - total length of all other chunks
        // - 8 + data size for sample data
        let mut size = SIZE_WAVE_HEADER + SIZE_FMT_CHUNK;
        for c in data.get_chunks() {
            size += SIZE_CHUNK_HEADER + c.get_num_bytes();
        }
        size += SIZE_CHUNK_HEADER + data.get_num_bytes() as u32;

        // Write RIFF header + size
        dest.write(&CID_RIFF.to_le_bytes())?;
        dest.write(&size.to_le_bytes())?;

        // Write WAVE header
        dest.write(&CID_WAVE.to_le_bytes())?;

        // Write fmt chunk
        WavHandler::write_chunk_object(dest, CID_FMT, data.get_fmt(), 16).unwrap();

        // Write any extra chunks
        for c in data.get_chunks() {
            WavHandler::write_chunk(dest, c.get_chunk_id(), &**c.get_data()).unwrap();
        }

        // Write data chunk
        dest.write(&CID_DATA.to_le_bytes())?;
        dest.write(&(data.get_samples().get_num_bytes() as u32).to_le_bytes())?;
        match &**data.get_samples() {
            WavDataType::PCM8(v) => WavHandler::write_samples(dest, &v, data.get_samples().get_num_bytes()),
            WavDataType::PCM16(v) => WavHandler::write_samples(dest, &v, data.get_samples().get_num_bytes()),
            WavDataType::FLOAT32(v) => WavHandler::write_samples(dest, &v, data.get_samples().get_num_bytes()),
            WavDataType::FLOAT64(v) => WavHandler::write_samples(dest, &v, data.get_samples().get_num_bytes()),
        }.unwrap();

        Ok(())
    }

    // Write a chunk to the output stream.
    fn write_chunk_object<W: Write, T: Sized>(dest: &mut W, cid: u32, data: &T, size: usize) -> Result<(), std::io::Error> {
        dest.write(&cid.to_le_bytes())?;
        dest.write(&(size as u32).to_le_bytes())?;
        unsafe {
            let data_slice = ::std::slice::from_raw_parts((data as *const _) as *const u8, size);
            dest.write(data_slice)?;
        }
        Ok(())
    }

    // Write a chunk to the output stream.
    fn write_chunk<W: Write>(dest: &mut W, cid: u32, data: &[u8]) -> Result<(), std::io::Error> {
        dest.write(&cid.to_le_bytes())?;
        dest.write(&(data.len() as u32).to_le_bytes())?;
        dest.write(data)?;
        Ok(())
    }

    // Write the sample data to the output stream.
    //
    // TODO: This currently directly writes the sample data, without byte order
    // conversion. Might need an update to work on other architectures.
    fn write_samples<W: Write, T: Sized>(dest: &mut W, data: &[T], num_bytes: usize) -> Result<(), std::io::Error> {
        unsafe {
            let samples = ::std::slice::from_raw_parts((data as *const _) as *const u8, num_bytes);
            dest.write(samples)?;
        }
        // Check if we need to add one byte for padding
        if (num_bytes & 0x01) == 0x01 {
            dest.write(&[0x00])?;
        }
        Ok(())
    }
}

// ----------------------------------------------
//                  Unit tests
// ----------------------------------------------

struct TestContext {
}

impl TestContext {
    pub fn new() -> Self {
        TestContext{}
    }

    pub fn test_read(&mut self, ptr: &[u8]) -> bool {
        let result = self.get_data(ptr);
        match result {
            Ok(_) => true,
            Err(()) => false
        }
    }

    pub fn get_data(&mut self, ptr: &[u8]) -> Result<WavData, ()> {
        use std::io::Cursor;
        let reader = Cursor::new(ptr);
        WavHandler::read_content(reader)
    }

    pub fn test_write(&mut self, samples: Box<WavDataType>, expected: &[u8]) -> bool {
        let data = WavData::new_from_data(samples);
        let mut buffer = Vec::new();
        let result = self.put_data(&mut buffer, &data);
        match result {
            Ok(_) => return buffer == expected,
            Err(_) => return false,
        }
    }

    pub fn put_data(&mut self, target: &mut Vec<u8>, data: &WavData) -> Result<(), std::io::Error> {
        WavHandler::write_content(target, data)
    }
}

#[test]
fn incomplete_riff_id_is_rejected() {
    let mut context = TestContext::new();

    let incomplete_riff: &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8,
    ];

    assert!(context.test_read(incomplete_riff) == false);
}

#[test]
fn empty_riff_is_rejected() {
    let mut context = TestContext::new();

    let empty_riff: &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x00, 0x00, 0x00, 0x00,
    ];

    assert!(context.test_read(empty_riff) == false);
}

#[test]
fn missing_riff_id_is_rejected() {
    let mut context = TestContext::new();

    let missing_riff_id : &[u8] = &[
        // RIFF header - invalid
        'R' as u8, 'x' as u8, 'x' as u8, 'x' as u8,
        0x04, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
    ];

    assert!(context.test_read(missing_riff_id) == false);
}

#[test]
fn missing_wave_id_is_rejected() {
    let mut context = TestContext::new();

    let missing_wave_id : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x04, 0x00, 0x00, 0x00,
        // Missing WAVE file ID
        'W' as u8, 'O' as u8, 'V' as u8, 'E' as u8,
    ];

    assert!(context.test_read(missing_wave_id) == false);
}

#[test]
fn incomplete_wave_id_is_rejected() {
    let mut context = TestContext::new();

    let incomplete_wave_id : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x03, 0x00, 0x00, 0x00,
        // Missing WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8,
    ];

    assert!(context.test_read(incomplete_wave_id) == false);
}

#[test]
fn valid_riff_empty_wave_is_rejected() {
    let mut context = TestContext::new();

    let empty_wave: &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x04, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
    ];

    assert!(context.test_read(empty_wave) == false);
}

#[test]
fn single_sample_byte_can_be_read() {
    let mut context = TestContext::new();

    let single_sample : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x04, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // fmt chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0x12, 0x00, 0x00, 0x00,
        0x01, 0x00,             // PCM
        0x01, 0x00,             // 1 channel
        0x44, 0xAC, 0x00, 0x00, // 44100 Hz
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
        0x08, 0x00,             // 8 bit per sample
        0x00, 0x00,
        // data chunk
        'd' as u8, 'a' as u8, 't' as u8, 'a' as u8,
        0x01, 0x00, 0x00, 0x00,
        0x42 // Single byte
    ];

    let result = context.get_data(single_sample);
    if let Ok(wav_file) = result {
        assert!(wav_file.get_num_bytes() == 1);
        if let WavDataType::PCM8(v) = &**wav_file.get_samples() {
            assert!(v[0] == 0x42);
        } else {
            assert!(false);
        }
    } else {
        assert!(false);
    }
}

#[test]
fn incomplete_chunk_is_rejected() {
    let mut context = TestContext::new();

    let incomplete_chunk: &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x04, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // data chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0x02, 0x00, 0x00, 0x00, // Size = 2
        0x42                    // Only single byte of data
    ];

    assert!(context.test_read(incomplete_chunk) == false);
}

#[test]
fn invalid_size_is_handled() {
    let mut context = TestContext::new();

    let incomplete_chunk: &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x04, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // data chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0xff, 0xff, 0xff, 0xff, // Size = 0xFFFFFFFF
        0x42                    // Only single byte of data
    ];

    assert!(context.test_read(incomplete_chunk) == false);
}

#[test]
fn unknown_chunks_are_skipped() {
    let mut context = TestContext::new();

    let single_sample : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x04, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // unknown chunk
        'n' as u8, 'u' as u8, 'l' as u8, 'l' as u8,
        0x01, 0x00, 0x00, 0x00,
        0xFF,
        // fmt chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0x12, 0x00, 0x00, 0x00,
        0x01, 0x00,             // PCM
        0x01, 0x00,             // 1 channel
        0x44, 0xAC, 0x00, 0x00, // 44100 Hz
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
        0x10, 0x00,             // 16 bit per sample
        0x00, 0x00,
        // data chunk
        'd' as u8, 'a' as u8, 't' as u8, 'a' as u8,
        0x02, 0x00, 0x00, 0x00,
        0x42, 0x43
    ];

    let result = context.get_data(single_sample);
    if let Ok(wav_file) = result {
        assert!(wav_file.get_num_bytes() == 2);
    } else {
        assert!(false);
    }
}

#[test]
fn u8_can_be_read() {
    let mut context = TestContext::new();

    let single_sample : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x04, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // fmt chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0x12, 0x00, 0x00, 0x00,
        0x01, 0x00,             // PCM
        0x01, 0x00,             // 1 channel
        0x44, 0xAC, 0x00, 0x00, // 44100 Hz
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
        0x08, 0x00,             // 8 bit per sample
        0x00, 0x00,
        // data chunk
        'd' as u8, 'a' as u8, 't' as u8, 'a' as u8,
        0x01, 0x00, 0x00, 0x00,
        0x42,
    ];

    let result = context.get_data(single_sample);
    if let Ok(wav_file) = result {
        assert!(wav_file.get_num_bytes() == 1);
        if let WavDataType::PCM8(data) = &**wav_file.get_samples() {
            assert!(data.len() == 1);
            assert!(data[0] == 0x42_u8);
        } else {
            assert!(false);
        }
    } else {
        assert!(false);
    }
}

#[test]
fn u16_can_be_read() {
    let mut context = TestContext::new();

    let single_sample : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x04, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // fmt chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0x12, 0x00, 0x00, 0x00,
        0x01, 0x00,             // PCM
        0x01, 0x00,             // 1 channel
        0x44, 0xAC, 0x00, 0x00, // 44100 Hz
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
        0x10, 0x00,             // 16 bit per sample
        0x00, 0x00,
        // data chunk
        'd' as u8, 'a' as u8, 't' as u8, 'a' as u8,
        0x02, 0x00, 0x00, 0x00,
        0x12, 0x34
    ];

    let result = context.get_data(single_sample);
    if let Ok(wav_file) = result {
        assert!(wav_file.get_num_bytes() == 2);
        if let WavDataType::PCM16(data) = &**wav_file.get_samples() {
            assert!(data.len() == 1);
            assert!(data[0] == 0x3412_i16);
        } else {
            assert!(false);
        }
    } else {
        assert!(false);
    }
}

#[test]
fn f32_can_be_read() {
    let mut context = TestContext::new();

    let single_sample : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x04, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // fmt chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0x12, 0x00, 0x00, 0x00,
        0x03, 0x00,             // Float
        0x01, 0x00,             // 1 channel
        0x44, 0xAC, 0x00, 0x00, // 44100 Hz
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
        0x20, 0x00,             // 32 bit per sample
        0x00, 0x00,
        // data chunk
        'd' as u8, 'a' as u8, 't' as u8, 'a' as u8,
        0x04, 0x00, 0x00, 0x00,
        0xb6, 0xf3, 0x9d, 0x3f  // = 1.234 in LE format
    ];

    let result = context.get_data(single_sample);
    if let Ok(wav_file) = result {
        assert!(wav_file.get_num_bytes() == 4);
        if let WavDataType::FLOAT32(data) = &**wav_file.get_samples() {
            assert!(data.len() == 1);
            assert!(data[0] == 1.234_f32);
        } else {
            assert!(false);
        }
    } else {
        assert!(false);
    }
}

#[test]
fn f64_can_be_read() {
    let mut context = TestContext::new();

    let single_sample : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x04, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // fmt chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0x12, 0x00, 0x00, 0x00,
        0x03, 0x00,             // Float
        0x01, 0x00,             // 1 channel
        0x44, 0xAC, 0x00, 0x00, // 44100 Hz
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
        0x40, 0x00,             // 64 bit per sample
        0x00, 0x00,
        // data chunk
        'd' as u8, 'a' as u8, 't' as u8, 'a' as u8,
        0x08, 0x00, 0x00, 0x00,
        0x58, 0x39, 0xb4, 0xc8,  // = 1.234 in LE format.
        0x76, 0xbe, 0xf3, 0x3f
    ];

    let result = context.get_data(single_sample);
    if let Ok(wav_file) = result {
        assert!(wav_file.get_num_bytes() == 8);
        if let WavDataType::FLOAT64(data) = &**wav_file.get_samples() {
            assert!(data.len() == 1);
            assert!(data[0] == 1.234_f64);
        } else {
            assert!(false);
        }
    } else {
        assert!(false);
    }
}

fn show_data(data: &[u8]) {
    let mut child = Command::new("xxd").stdin(Stdio::piped()).spawn().unwrap();
    let child_stdin = child.stdin.as_mut().unwrap();
    child_stdin.write_all(data).unwrap();
}

#[test]
fn u8_can_be_written() {
    let mut context = TestContext::new();
    let samples = Box::new(WavDataType::PCM8(vec![0, 1, 2, 3]));
    let expected : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x28, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // fmt chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0x10, 0x00, 0x00, 0x00,
        0x01, 0x00,             // PCM
        0x01, 0x00,             // 1 channel
        0x44, 0xAC, 0x00, 0x00, // 44100 Hz
        0x44, 0xAC, 0x00, 0x00, // Avg data rate
        0x00, 0x00,             // Block align
        0x08, 0x00,             // 8 bit per sample
        // data chunk
        'd' as u8, 'a' as u8, 't' as u8, 'a' as u8,
        0x04, 0x00, 0x00, 0x00, // 4 u8 = 4 bytes
        0x00, 0x01, 0x02, 0x03,
    ];
    assert!(context.test_write(samples, expected));
}

#[test]
fn s16_can_be_written() {
    let mut context = TestContext::new();
    let samples = Box::new(WavDataType::PCM16(vec![-1, 0, 1, 2]));
    let expected : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x2C, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // fmt chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0x10, 0x00, 0x00, 0x00,
        0x01, 0x00,             // PCM
        0x01, 0x00,             // 1 channel
        0x44, 0xAC, 0x00, 0x00, // 44100 Hz
        0x88, 0x58, 0x01, 0x00, // Avg data rate
        0x00, 0x00,             // Block align
        0x10, 0x00,             // 16 bit per sample
        // data chunk
        'd' as u8, 'a' as u8, 't' as u8, 'a' as u8,
        0x08, 0x00, 0x00, 0x00, // 4 s16 = 8 bytes
        0xFF, 0xFF, 0x00, 0x00,
        0x01, 0x00, 0x02, 0x00,
    ];
    assert!(context.test_write(samples, expected));
}

#[test]
fn f32_can_be_written() {
    let mut context = TestContext::new();
    let samples = Box::new(WavDataType::FLOAT32(vec![0.0, 0.1, 0.2, 0.3]));
    let expected : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x34, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // fmt chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0x10, 0x00, 0x00, 0x00,
        0x03, 0x00,             // Float
        0x01, 0x00,             // 1 channel
        0x44, 0xAC, 0x00, 0x00, // 44100 Hz
        0x10, 0xb1, 0x02, 0x00, // Avg data rate
        0x00, 0x00,             // Block align
        0x20, 0x00,             // 32 bit per sample
        // data chunk
        'd' as u8, 'a' as u8, 't' as u8, 'a' as u8,
        0x10, 0x00, 0x00, 0x00, // 4 f32 = 16 bytes
        0x00, 0x00, 0x00, 0x00, // 0.0
        0xcd, 0xcc, 0xcc, 0x3d, // 0.1
        0xcd, 0xcc, 0x4c, 0x3e, // 0.2
        0x9a, 0x99, 0x99, 0x3e, // 0.3
    ];
    assert!(context.test_write(samples, expected));
}

#[test]
fn f64_can_be_written() {
    let mut context = TestContext::new();
    let samples = Box::new(WavDataType::FLOAT64(vec![0.1]));
    let expected : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x2c, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // fmt chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0x10, 0x00, 0x00, 0x00,
        0x03, 0x00,             // Float
        0x01, 0x00,             // 1 channel
        0x44, 0xAC, 0x00, 0x00, // 44100 Hz
        0x20, 0x62, 0x05, 0x00, // Avg data rate
        0x00, 0x00,             // Block align
        0x40, 0x00,             // 64 bit per sample
        // data chunk
        'd' as u8, 'a' as u8, 't' as u8, 'a' as u8,
        0x08, 0x00, 0x00, 0x00, // 1 f64 = 8 bytes
        0x9a, 0x99, 0x99, 0x99, // 0.0
        0x99, 0x99, 0xb9, 0x3f,
    ];
    assert!(context.test_write(samples, expected));
}

#[test]
fn odd_number_of_u8_is_padded() {
    let mut context = TestContext::new();
    let samples = Box::new(WavDataType::PCM8(vec![1, 2, 3]));
    let expected : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x27, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // fmt chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0x10, 0x00, 0x00, 0x00,
        0x01, 0x00,             // PCM
        0x01, 0x00,             // 1 channel
        0x44, 0xAC, 0x00, 0x00, // 44100 Hz
        0x44, 0xAC, 0x00, 0x00, // Avg data rate
        0x00, 0x00,             // Block align
        0x08, 0x00,             // 8 bit per sample
        // data chunk
        'd' as u8, 'a' as u8, 't' as u8, 'a' as u8,
        0x03, 0x00, 0x00, 0x00, // 3 u8 = 3 bytes
        0x01, 0x02, 0x03, 0x00
    ];
    assert!(context.test_write(samples, expected));
}

#[test]
fn custom_chunk_can_be_added() {
    let cid_test: u32 = 0x54534554;
    let mut context = TestContext::new();
    let samples = Box::new(WavDataType::PCM8(vec![1, 2, 3]));
    let mut wav_data = WavData::new_from_data(samples);
    let chunk_data = Box::new(vec![0x05, 0x06, 0x07, 0x08]);
    let chunk = Chunk::new_from_data(cid_test, chunk_data);
    wav_data.add_chunk(chunk);
    let expected : &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x33, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // fmt chunk
        'f' as u8, 'm' as u8, 't' as u8, ' ' as u8,
        0x10, 0x00, 0x00, 0x00,
        0x01, 0x00,             // PCM
        0x01, 0x00,             // 1 channel
        0x44, 0xAC, 0x00, 0x00, // 44100 Hz
        0x44, 0xAC, 0x00, 0x00, // Avg data rate
        0x00, 0x00,             // Block align
        0x08, 0x00,             // 8 bit per sample
        // New chunk goes here
        0x54, 0x45, 0x53, 0x54, // CID = TEST
        0x04, 0x00, 0x00, 0x00, // size
        0x05, 0x06, 0x07, 0x08, // data
        // data chunk
        'd' as u8, 'a' as u8, 't' as u8, 'a' as u8,
        0x03, 0x00, 0x00, 0x00, // 3 u8 = 3 bytes
        0x01, 0x02, 0x03, 0x00
    ];
    let mut buffer = Vec::new();
    let result = context.put_data(&mut buffer, &wav_data);
    match result {
        Ok(()) => {
            assert!(buffer == expected);
        },
        Err(_) => {
            assert!(false);
        },
    }
}
