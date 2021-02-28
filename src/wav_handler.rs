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

// Format tag identifiers (don't care about uLaw for now)
const FMT_PCM: u16 = 1;
const FMT_FLOAT: u16 = 3;

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

/// Represents the format chunk that needs to be present in every WAV file.
#[repr(C, packed)]
#[derive(Debug, Default, Copy, Clone)]
pub struct FmtChunk {
    pub format_tag: u16,      // wFormatTag      2   Format code
    pub num_channels: u16,    // nChannels       2   Number of interleaved channels
    pub sample_rate: u32,     // nSamplesPerSec  4   Sampling rate (blocks per second)
    pub avg_data_rate: u32,   // nAvgBytesPerSec 4   Data rate
    pub block_align: u16,     // nBlockAlign     2   Data block size (bytes)
    pub bits_per_sample: u16, // wBitsPerSample  2   Bits per sample
    pub cb_size: u16,         // cbSize          2   Size of the extension (0 or 22)
    pub valid_bits: u16,      // wValidBitsPerSample 2   Number of valid bits
    pub channel_mask: u32,    // dwChannelMask   4   Speaker position mask
    pub sub_format: [u8; 16]  // SubFormat       16  GUID, including the data format code
}

impl FmtChunk {
    pub fn new(data: &WavDataType) -> FmtChunk {
        let bps = data.get_bits_per_sample();
        FmtChunk{
            format_tag: data.get_format_tag(),
            num_channels: 1,
            sample_rate: 44100,
            avg_data_rate: 44100 * (bps as u32 / 8),
            block_align: 0,
            bits_per_sample: bps,
            cb_size: 0,
            valid_bits: 0,
            channel_mask: 0,
            sub_format: [0u8; 16],
        }
    }

    /// Get the number of audio channels defined in the WAV file.
    pub fn get_num_channels(&self) -> usize {
        self.num_channels as usize
    }

    /// Get the number of bits per sample defined in the WAV file.
    pub fn get_bits_per_sample(&self) -> usize {
        self.bits_per_sample as usize
    }
}

// Chunk containing the sample data
struct DataChunk {
    size: usize,
    data: Box<WavDataType>
}

impl Default for DataChunk {
    fn default() -> Self {
        DataChunk{size: 0, data: Box::new(WavDataType::PCM16(vec!()))}
    }
}

// Generic chunk
pub struct Chunk {
    chunk_id: u32,
    size: u32, // Size of data, excluding the header
    data: Box<Vec<u8>>,
}

impl Chunk {
    pub fn new(chunk_id: u32, size: u32) -> Chunk {
        Chunk{chunk_id, size, data: Box::new(vec![0_u8; size as usize])}
    }
}

/// Container for the different sample data types.
#[derive(Debug)]
pub enum WavDataType {
    PCM8(Vec<u8>),
    PCM16(Vec<i16>),
    FLOAT32(Vec<f32>),
    FLOAT64(Vec<f64>)
}

impl WavDataType {
    /// Get the number of samples in the container.
    pub fn get_num_samples(&self) -> usize {
        match self {
            WavDataType::PCM8(v) => v.len(),
            WavDataType::PCM16(v) => v.len(),
            WavDataType::FLOAT32(v) => v.len(),
            WavDataType::FLOAT64(v) => v.len(),
        }
    }

    /// Get a printable representation of the data type.
    pub fn get_type(&self) -> &str {
        match self {
            WavDataType::PCM8(_) => "PCM8",
            WavDataType::PCM16(_) => "PCM16",
            WavDataType::FLOAT32(_) => "Float32",
            WavDataType::FLOAT64(_) => "Float64",
        }
    }

    /// Get the format tag of the FMT chunk that represents the current
    /// data type.
    pub fn get_format_tag(&self) -> u16 {
        match self {
            WavDataType::PCM8(_) => FMT_PCM,
            WavDataType::PCM16(_) => FMT_PCM,
            WavDataType::FLOAT32(_) => FMT_FLOAT,
            WavDataType::FLOAT64(_) => FMT_FLOAT,
        }
    }

    /// Get the format tag of the FMT chunk that represents the current
    /// data type.
    pub fn get_bits_per_sample(&self) -> u16 {
        match self {
            WavDataType::PCM8(_) => 8,
            WavDataType::PCM16(_) => 16,
            WavDataType::FLOAT32(_) => 32,
            WavDataType::FLOAT64(_) => 64,
        }
    }

    pub fn get_num_bytes(&self) -> usize {
        self.get_num_samples() * (self.get_bits_per_sample() / 8) as usize
    }
}

/// Contains the format information and sample data read from the file.
pub struct WavData {
    info: FmtChunk,
    data: DataChunk,
    chunks: Vec<Chunk>,
}

// Macro to append sample data to existing wave data if type matches
macro_rules! extend_samples {
    ($target:ident, $self:ident, $samples:ident, $x:ident::$y:ident) => {
        {
            if let $x::$y(s) = *$samples {
                let size = s.len();
                $target.extend(s);
                $self.data.size = $target.len() * ($self.info.bits_per_sample / 8) as usize;
                println!("Added {} samples, new size: {} with {} samples",
                    size, $self.get_data_size(), $self.get_num_samples());
            } else {
                return Err(());
            }
        }
    }
}

impl WavData {
    pub fn new() -> WavData {
        WavData{
            info: FmtChunk{..Default::default()},
            data: DataChunk{..Default::default()},
            chunks: vec!{}}
    }

    /// Create a struct containing the given list of samples.
    pub fn new_from_data(samples: Box<WavDataType>) -> WavData {
        let info = FmtChunk::new(&samples);
        let data = DataChunk{size: samples.get_num_bytes(), data: samples};
        let chunks = vec!{};
        WavData{
            info,
            data,
            chunks}
    }

    /// Append the given samples to the already existing samples.
    pub fn append_samples(&mut self, samples: Box<WavDataType>) -> Result<(), ()> {
        match &mut **self.get_data_mut() {
            WavDataType::PCM8(ref mut v) => extend_samples!(v, self, samples, WavDataType::PCM8),
            WavDataType::PCM16(ref mut v) => extend_samples!(v, self, samples, WavDataType::PCM16),
            WavDataType::FLOAT32(ref mut v) => extend_samples!(v, self, samples, WavDataType::FLOAT32),
            WavDataType::FLOAT64(ref mut v) => extend_samples!(v, self, samples, WavDataType::FLOAT64),
        }
        Ok(())
    }

    /// Add an arbitraty chunk to the list of chunks.
    pub fn add_chunk(&mut self, data: Chunk) {
        self.chunks.push(data);
    }

    /// Get the FMT chunk.
    pub fn get_fmt(&self) -> &FmtChunk {
        &self.info
    }

    /// Get mutable reference to the FMT chunk.
    pub fn get_fmt_mut(&mut self) -> &mut FmtChunk {
        &mut self.info
    }

    pub fn get_num_samples(&self) -> usize {
        self.get_data().get_num_samples()
    }

    /// Get the number of sample bytes.
    pub fn get_data_size(&self) -> usize {
        self.data.size
    }

    /// Get the vector with sample data.
    pub fn get_data(&self) -> &Box<WavDataType> {
        return &self.data.data;
    }

    /// Get the vector with sample data.
    pub fn get_data_mut(&mut self) -> &mut Box<WavDataType> {
        return &mut self.data.data;
    }
}

// Trait to pack the vector with sample data into a matching enum value
trait WavDataTypeGetter<T> {
    fn get_wav_data_type(&self, data: Vec<T>) -> WavDataType;
}

impl WavDataTypeGetter<u8> for u8 {
    fn get_wav_data_type(&self, data: Vec<u8>) -> WavDataType {
        WavDataType::PCM8(data)
    }
}

impl WavDataTypeGetter<i16> for i16 {
    fn get_wav_data_type(&self, data: Vec<i16>) -> WavDataType {
        WavDataType::PCM16(data)
    }
}

impl WavDataTypeGetter<f32> for f32 {
    fn get_wav_data_type(&self, data: Vec<f32>) -> WavDataType {
        WavDataType::FLOAT32(data)
    }
}

impl WavDataTypeGetter<f64> for f64 {
    fn get_wav_data_type(&self, data: Vec<f64>) -> WavDataType {
        WavDataType::FLOAT64(data)
    }
}

pub struct WavHandler;

/// Handles reading and writing of .wav files.
///
/// Reads wave files into memory as vectors of samples. The resulting struct
/// contains the FMT info, the data, and a list of additional chunks that
/// were found in the file (TBD).
///
/// TODO:
/// - Store additional chunks in vector
/// - Implement writing data to file
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
                        file.info = WavHandler::read_chunk(&mut source, header.size as usize)?;
                        fmt_found = true;
                    }
                    CID_DATA => {
                        // Found data section, read samples
                        file.data = WavHandler::read_samples(&mut source,
                                                             header.size as usize,
                                                             file.info.format_tag,
                                                             file.info.bits_per_sample)?;
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
    fn read_chunk<R: Read, T: std::fmt::Debug>(source: &mut R, size: usize) -> Result<T, ()> {
        let mut data: T = unsafe { mem::zeroed() };
        unsafe {
            let data_slize = std::slice::from_raw_parts_mut(&mut data as *mut _ as *mut u8, size);
            if let Err(_) = source.read_exact(data_slize) {
                error!("Reading chunk data failed");
                return Err(());
            }
            debug!("Read chunk: {:#?}", data);
        }
        Ok(data)
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
            Ok(s) => Ok(DataChunk{
                        size: num_bytes, 
                        data: Box::new(s)
                    }),
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
        for c in &data.chunks {
            size += SIZE_CHUNK_HEADER + c.size;
        }
        size += SIZE_CHUNK_HEADER + data.get_data_size() as u32;

        // Write RIFF header + size
        dest.write(&CID_RIFF.to_le_bytes())?;
        dest.write(&size.to_le_bytes())?;

        // Write WAVE header
        dest.write(&CID_WAVE.to_le_bytes())?;

        // Write fmt chunk
        WavHandler::write_chunk(dest, CID_FMT, &data.info, 16).unwrap();

        // Write any extra chunks
        for c in &data.chunks {
            WavHandler::write_chunk(dest, c.chunk_id, &c.data, c.size as usize).unwrap();
        }

        // Write data chunk
        // TODO: Clean up the names (data.data.data...)
        dest.write(&CID_DATA.to_le_bytes())?;
        dest.write(&(data.data.size as u32).to_le_bytes())?;
        match &*data.data.data {
            WavDataType::PCM8(v) => WavHandler::write_samples(dest, &v, data.data.size),
            WavDataType::PCM16(v) => WavHandler::write_samples(dest, &v, data.data.size),
            WavDataType::FLOAT32(v) => WavHandler::write_samples(dest, &v, data.data.size),
            WavDataType::FLOAT64(v) => WavHandler::write_samples(dest, &v, data.data.size),
        }.unwrap();

        Ok(())
    }

    // Write a chunk to the output stream.
    fn write_chunk<W: Write, T: Sized>(dest: &mut W, cid: u32, data: &T, size: usize) -> Result<(), std::io::Error> {
        dest.write(&cid.to_le_bytes())?;
        dest.write(&(size as u32).to_le_bytes())?;
        unsafe {
            let data_slice = ::std::slice::from_raw_parts((data as *const _) as *const u8, size);
            dest.write(data_slice)?;
        }
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

    pub fn test(&mut self, ptr: &[u8]) -> bool {
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

    pub fn put_data(&mut self, samples: Box<WavDataType>) -> Vec<u8> {
        let data = WavData::new_from_data(samples);
        let mut buffer = Vec::new();
        WavHandler::write_content(&mut buffer, &data).unwrap();
        buffer
    }
}

#[test]
fn incomplete_riff_id_is_rejected() {
    let mut context = TestContext::new();

    let incomplete_riff: &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8,
    ];

    assert!(context.test(incomplete_riff) == false);
}

#[test]
fn empty_riff_is_rejected() {
    let mut context = TestContext::new();

    let empty_riff: &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x00, 0x00, 0x00, 0x00,
    ];

    assert!(context.test(empty_riff) == false);
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

    assert!(context.test(missing_riff_id) == false);
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

    assert!(context.test(missing_wave_id) == false);
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

    assert!(context.test(incomplete_wave_id) == false);
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

    assert!(context.test(empty_wave) == false);
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
        assert!(wav_file.get_data_size() == 1);
        if let WavDataType::PCM8(v) = &**wav_file.get_data() {
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

    assert!(context.test(incomplete_chunk) == false);
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

    assert!(context.test(incomplete_chunk) == false);
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
        assert!(wav_file.get_data_size() == 2);
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
        assert!(wav_file.get_data_size() == 1);
        if let WavDataType::PCM8(data) = &**wav_file.get_data() {
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
        assert!(wav_file.get_data_size() == 2);
        if let WavDataType::PCM16(data) = &**wav_file.get_data() {
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
        assert!(wav_file.get_data_size() == 4);
        if let WavDataType::FLOAT32(data) = &**wav_file.get_data() {
            assert!(data.len() == 1);
            println!("Value: {}", data[0]);
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
        assert!(wav_file.get_data_size() == 8);
        if let WavDataType::FLOAT64(data) = &**wav_file.get_data() {
            assert!(data.len() == 1);
            println!("Value: {}", data[0]);
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
    let result = context.put_data(samples);
    assert!(result == expected);
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
    let result = context.put_data(samples);
    assert!(result == expected);
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
    let result = context.put_data(samples);
    assert!(result == expected);
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
    let result = context.put_data(samples);
    assert!(result == expected);
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
    let result = context.put_data(samples);
    //show_data(&result);
    assert!(result == expected);
}

#[test]
fn samples_can_be_appended_to_u8() {
    let samples_1 = Box::new(WavDataType::PCM8(vec![1, 2, 3]));
    let samples_2 = Box::new(WavDataType::PCM8(vec![4, 5, 6]));
    let mut data = WavData::new_from_data(samples_1);
    data.append_samples(samples_2).unwrap();
    assert!(data.get_num_samples() == 6);
    assert!(data.get_data_size() == 6);
}

#[test]
fn samples_can_be_appended_to_i16() {
    let samples_1 = Box::new(WavDataType::PCM16(vec![1, 2, 3]));
    let samples_2 = Box::new(WavDataType::PCM16(vec![4, 5, 6]));
    let mut data = WavData::new_from_data(samples_1);
    data.append_samples(samples_2).unwrap();
    assert!(data.get_num_samples() == 6);
    assert!(data.get_data_size() == 12);
}

#[test]
fn samples_can_be_appended_to_f32() {
    let samples_1 = Box::new(WavDataType::FLOAT32(vec![1.0, 2.0, 3.0]));
    let samples_2 = Box::new(WavDataType::FLOAT32(vec![4.0, 5.0, 6.0]));
    let mut data = WavData::new_from_data(samples_1);
    data.append_samples(samples_2).unwrap();
    assert!(data.get_num_samples() == 6);
    assert!(data.get_data_size() == 24);
}

#[test]
fn samples_can_be_appended_to_f64() {
    let samples_1 = Box::new(WavDataType::FLOAT64(vec![1.0, 2.0, 3.0]));
    let samples_2 = Box::new(WavDataType::FLOAT64(vec![4.0, 5.0, 6.0]));
    let mut data = WavData::new_from_data(samples_1);
    data.append_samples(samples_2).unwrap();
    assert!(data.get_num_samples() == 6);
    assert!(data.get_data_size() == 48);
}

#[test]
fn append_data_missmatch_is_detected() {
    let samples_1 = Box::new(WavDataType::FLOAT64(vec![1.0, 2.0, 3.0]));
    let samples_2 = Box::new(WavDataType::FLOAT32(vec![4.0, 5.0, 6.0]));
    let mut data = WavData::new_from_data(samples_1);
    let result = data.append_samples(samples_2);
    assert!(matches!(result, Err(_)));
}

#[test]
fn custom_chunk_can_be_added() {
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
    let result = context.put_data(samples);
    //show_data(&result);
    assert!(result == expected);
}
