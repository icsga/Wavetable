use std::fs::File;
use std::io::{Read, BufReader};
use std::mem;

use log::{debug, error, info, trace};

// List of Chunk IDs as u32 values (little endian)
// TODO: Check how to improve this for portability
const CID_RIFF: u32 = 0x46464952;
const CID_WAVE: u32 = 0x45564157;
const CID_FMT:  u32 = 0x20746d66;
const CID_DATA: u32 = 0x61746164;

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
struct ChunkHeader {
    chunk_id: u32,
    size: u32
}

// Generic chunk
struct Chunk {
    chunk_id: u32,
    size: u32,
    data: Box<Vec<u8>>,
}

#[repr(C, packed)]
#[derive(Debug, Default, Copy, Clone)]
struct FmtChunk {
    format_tag: u16,      // wFormatTag      2   Format code
    num_channels: u16,    // nChannels       2   Number of interleaved channels
    sample_rate: u32,     // nSamplesPerSec  4   Sampling rate (blocks per second)
    avg_data_rate: u32,   // nAvgBytesPerSec 4   Data rate
    block_align: u16,     // nBlockAlign     2   Data block size (bytes)
    bits_per_sample: u16, // wBitsPerSample  2   Bits per sample
    cb_size: u16,         // cbSize          2   Size of the extension (0 or 22)
    valid_bits: u16,      // wValidBitsPerSample 2   Number of valid bits
    channel_mask: u32,    // dwChannelMask   4   Speaker position mask
    sub_format: [u8; 16]  // SubFormat       16  GUID, including the data format code
}

struct DataChunk {
    size: usize,
    data: Box<Vec<u8>>
}

impl Default for DataChunk {
    fn default() -> Self {
        DataChunk{size: 0, data: Box::new(vec!())}
    }
}

enum FileData {
    FdFmt(FmtChunk),
    FdData(DataChunk),
}

pub struct WavFile {
    info: FmtChunk,
    data: DataChunk,
    chunks: Vec<Chunk>,
}

impl WavFile {
    pub fn new() -> WavFile {
        WavFile{
            info: FmtChunk{..Default::default()},
            data: DataChunk{..Default::default()},
            chunks: vec!{}}
    }

    fn add_chunk(&mut self, data: Chunk) {
        self.chunks.push(data);
    }

    pub fn get_data_size(&self) -> usize {
        self.data.size
    }

    pub fn get_data(&self) -> &Box<Vec<u8>> {
        return &self.data.data;
    }
}

pub struct WavHandler {
}

/// Handles reading and writing of .wav files.
///
/// Reads wave files into memory as vectors of u8. The resulting struct
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
    /// let wavetable = WavHandler::read_file("test")?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_file(filename: &str) -> Result<WavFile, ()> {
        if filename == "" {
            return Err(());
        }
        debug!("Trying to read file [{}]", filename);
        let result = File::open(filename);
        if let Ok(file) = result {
            trace!("File [{}] opened, reading data", filename);
            let reader = BufReader::new(file);
            WavHandler::read_content(reader)
        } else {
            error!("Unable to open file [{}]", filename);
            Err(())
        }
    }

    /// Read contents of a wave file from the provided input stream.
    ///
    /// Source is any stream object implementing the Read trait.
    ///
    /// ``` no_run
    /// use wavetable::WavHandler;
    /// use std::io::BufReader;
    ///
    /// # fn main() -> Result<(), ()> {
    ///
    /// let reader = WavHandler::new();
    /// let data: &[u8] = &[0x00]; // Some buffer with wave data
    /// let buffer = BufReader::new(data);
    /// let wavedata = WavHandler::read_content(buffer)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_content<R: Read>(mut source: R) -> Result<WavFile, ()> {
        // Read RIFF header and filetype
        let result = WavHandler::read_riff_container(&mut source, CID_WAVE);
        let size = match result {
            Ok(s) => s,
            Err(()) => return Err(()),
        };

        let mut file = WavFile::new();
        let mut bytes_read: usize = 4; // Already read the 4 bytes of file type

        // Read chunks
        loop {
            let result = WavHandler::read_header(&mut source);
            if let Ok(header) = result {
                match header.chunk_id {
                    CID_FMT  => {
                        // Get data format
                        file.info = WavHandler::read_chunk(&mut source, header.size as usize)?;
                    }
                    CID_DATA => {
                        // Found data section, create wavetable
                        file.data = WavHandler::read_samples(&mut source, header.size as usize)?;
                    },
                    _ => WavHandler::skip_chunk(&mut source, header.size),
                }
                bytes_read += (header.size + 8) as usize;
            } else {
                break; // Error or finished reading file. In both cases return data read so far.
            }
        }
        if bytes_read == size {
            info!("Finished reading {} bytes", bytes_read);
        } else {
            error!("Invalid file size, read {} bytes, expected {}", bytes_read, size);
        }
        Ok(file)
    }

    // Read the RIFF container information from the input stream.
    //
    // This expects a RIFF header, followed by a 4-byte identifier (e.g.
    // "WAVE"), which is passed as argument.
    fn read_riff_container<R: Read>(source: &mut R, expected_cid: u32) -> Result<usize, ()> {
        // Read RIFF header
        let result = WavHandler::read_header(source);
        let data_size = match result {
            Ok(header) => {
                if header.chunk_id != CID_RIFF {
                    error!("Unexpected chunk ID, expected RIFF, found {}", WavHandler::get_id_name(header.chunk_id));
                    return Err(());
                }
                header.size as usize
            }
            Err(()) => return Err(()),
        };
        let result = WavHandler::read_id(source);
        match result {
            Ok(header) => {
                // RIFF header is followed by 4 bytes giving the file type
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

    // Read chunk ID, without the size information (for file type ID, e.g. "WAVE")
    fn read_id<R: Read>(source: &mut R) -> Result<u32, ()> {
        let mut header: u32 = 0;
        let header_size = 4;
        unsafe {
            let header_slize = std::slice::from_raw_parts_mut(&mut header as *mut _ as *mut u8, header_size);
            if let Err(_) = source.read_exact(header_slize) {
                error!("Reading chunk ID failed");
                return Err(());
            }
            info!("Read chunk ID: {} - {:x}", WavHandler::get_id_name(header), header);
        }
        Ok(header)
    }

    // Read the header of the next chunk from the input stream.
    fn read_header<R: Read>(source: &mut R) -> Result<ChunkHeader, ()> {
        let mut header: ChunkHeader = unsafe { mem::zeroed() };
        let header_size = mem::size_of::<ChunkHeader>();
        unsafe {
            let header_slize = std::slice::from_raw_parts_mut(&mut header as *mut _ as *mut u8, header_size);
            if let Err(_) = source.read_exact(header_slize) {
                // If we can't read a full header, we might simply have reached
                // the end of the file. Return Err to signal no header was read.
                return Err(());
            }
            info!("Read chunk ID: {} - {:x}, size {}", WavHandler::get_id_name(header.chunk_id), header.chunk_id, header.size);
        }
        Ok(header)
    }

    // Read the contents of a chunk from the input stream.
    //
    // The chunk header is assumed to have been read already.
    fn read_chunk<R: Read, T: std::fmt::Debug>(source: &mut R, size: usize) -> Result<T, ()> {
        let mut data: T = unsafe { mem::zeroed() };
        unsafe {
            let data_slize = std::slice::from_raw_parts_mut(&mut data as *mut _ as *mut u8, size);
            if let Err(_) = source.read_exact(data_slize) {
                error!("Reading chunk data failed");
                return Err(());
            }
            info!("Read chunk: {:#?}", data);
        }
        Ok(data)
    }

    // Read samples into buffer.
    fn read_samples<R: Read>(source: &mut R, num_bytes: usize) -> Result<DataChunk, ()> {
        let mut buff: Vec<u8> = vec![0u8; num_bytes];
        source.read_exact(&mut buff).unwrap();
        debug!("Read {} bytes of sample data", num_bytes);
        Ok(DataChunk{
            size: num_bytes, 
            data: Box::new(buff)
        })
    }

    // Skip over the rest of the current chunk to the next header.
    fn skip_chunk<R: Read>(source: &mut R, num_bytes: u32) {
        let mut buf: [u8; 1] = unsafe { mem::zeroed() };
        for _i in 0..num_bytes {
            source.read(&mut buf).unwrap();
        }
    }

    // Convert a given chunk ID from u32 to printable string.
    fn get_id_name(value: u32) -> String {
        let bytes = value.to_le_bytes();
        String::from_utf8(bytes.to_vec()).expect("Found invalid UTF-8")
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
        let reader = BufReader::new(ptr);
        let result = WavHandler::read_content(reader);
        match result {
            Ok(_) => true,
            Err(()) => false
        }
    }

    pub fn get_data(&mut self, ptr: &[u8]) -> Result<WavFile, ()> {
        let reader = BufReader::new(ptr);
        WavHandler::read_content(reader)
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
        // RIFF header - invalid
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
        // RIFF header - invalid
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x03, 0x00, 0x00, 0x00,
        // Missing WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8,
    ];

    assert!(context.test(incomplete_wave_id) == false);
}

#[test]
fn valid_riff_empty_wave_is_accepted() {
    let mut context = TestContext::new();

    let empty_wave: &[u8] = &[
        // RIFF header
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x04, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
    ];

    assert!(context.test(empty_wave));
}

#[test]
fn single_sample_byte_can_be_read() {
    let mut context = TestContext::new();

    let single_sample : &[u8] = &[
        // RIFF header - invalid
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x04, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // data chunk
        'd' as u8, 'a' as u8, 't' as u8, 'a' as u8,
        0x01, 0x00, 0x00, 0x00,
        0x42 // Single byte
    ];

    let result = context.get_data(single_sample);
    if let Ok(wav_file) = result {
        assert!(wav_file.get_data_size() == 1);
        assert!(wav_file.get_data()[0] == 0x42);
    } else {
        assert!(false);
    }
}

#[test]
fn incomplete_chunk_is_rejected() {
    let mut context = TestContext::new();

    let incomplete_chunk: &[u8] = &[
        // RIFF header - invalid
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
        // RIFF header - invalid
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
        // RIFF header - invalid
        'R' as u8, 'I' as u8, 'F' as u8, 'F' as u8,
        0x04, 0x00, 0x00, 0x00,
        // WAVE file ID
        'W' as u8, 'A' as u8, 'V' as u8, 'E' as u8,
        // unknown chunk
        'n' as u8, 'u' as u8, 'l' as u8, 'l' as u8,
        0x01, 0x00, 0x00, 0x00,
        0xFF,
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

