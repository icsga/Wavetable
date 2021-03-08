// Format tag identifiers (don't care about uLaw for now)
pub const FMT_PCM: u16 = 1;
pub const FMT_FLOAT: u16 = 3;

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
pub struct DataChunk {
    size: usize,
    data: Box<WavDataType>
}

impl DataChunk {
    pub fn new(size: usize, data: Box<WavDataType>) -> Self {
        Self{size, data}
    }

    pub fn get_num_bytes(&self) -> usize {
        self.size
    }

    pub fn get_data(&self) -> &Box<WavDataType> {
        &self.data
    }
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

    pub fn new_from_data(chunk_id: u32, data: Box<Vec<u8>>) -> Chunk {
        Chunk{chunk_id, size: data.len() as u32, data}
    }

    pub fn get_chunk_id(&self) -> u32 {
        self.chunk_id
    }

    pub fn get_num_bytes(&self) -> u32 {
        self.size
    }

    pub fn get_data(&self) -> &Box<Vec<u8>> {
        &self.data
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

// Trait to pack the vector with sample data into a matching enum value
pub trait WavDataTypeGetter<T> {
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

// Macro to append sample data to existing wave data if type matches
macro_rules! extend_samples {
    ($target:ident, $self:ident, $samples:ident, $enum:ident :: $variant:ident) => {
        {
            if let $enum::$variant(s) = *$samples {
                $target.extend(s);
                $self.data.size = $target.len() * ($self.info.bits_per_sample / 8) as usize;
            } else {
                return Err(());
            }
        }
    }
}

/// Contains the format information and sample data read from the file.
pub struct WavData {
    info: FmtChunk,
    data: DataChunk,
    chunks: Vec<Chunk>,
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

    /// Take ownership of the samples contained in the DataChunk.
    pub fn set_samples(&mut self, samples: DataChunk) {
        self.data = samples;
    }

    /// Append the given samples to the already existing samples.
    pub fn append_samples(&mut self, samples: Box<WavDataType>) -> Result<(), ()> {
        match &mut **self.get_samples_mut() {
            WavDataType::PCM8(ref mut v) => extend_samples!(v, self, samples, WavDataType::PCM8),
            WavDataType::PCM16(ref mut v) => extend_samples!(v, self, samples, WavDataType::PCM16),
            WavDataType::FLOAT32(ref mut v) => extend_samples!(v, self, samples, WavDataType::FLOAT32),
            WavDataType::FLOAT64(ref mut v) => extend_samples!(v, self, samples, WavDataType::FLOAT64),
        }
        Ok(())
    }

    /// Add an arbitrary chunk to the list of chunks.
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

    /// Get the number of samples.
    pub fn get_num_samples(&self) -> usize {
        self.get_samples().get_num_samples()
    }

    /// Get the number of sample bytes.
    pub fn get_num_bytes(&self) -> usize {
        self.data.size
    }

    /// Get the list of all extra chunks, not including FMT and DATA.
    pub fn get_chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    /// Get the vector with sample data.
    pub fn get_samples(&self) -> &Box<WavDataType> {
        return &self.data.data;
    }

    /// Get the vector with sample data as mutable reference.
    pub fn get_samples_mut(&mut self) -> &mut Box<WavDataType> {
        return &mut self.data.data;
    }
}

// ----------------------------------------------
//                  Unit tests
// ----------------------------------------------

#[test]
fn samples_can_be_appended_to_u8() {
    let samples_1 = Box::new(WavDataType::PCM8(vec![1, 2, 3]));
    let samples_2 = Box::new(WavDataType::PCM8(vec![4, 5, 6]));
    let mut data = WavData::new_from_data(samples_1);
    data.append_samples(samples_2).unwrap();
    assert!(data.get_num_samples() == 6);
    assert!(data.get_num_bytes() == 6);
}

#[test]
fn samples_can_be_appended_to_i16() {
    let samples_1 = Box::new(WavDataType::PCM16(vec![1, 2, 3]));
    let samples_2 = Box::new(WavDataType::PCM16(vec![4, 5, 6]));
    let mut data = WavData::new_from_data(samples_1);
    data.append_samples(samples_2).unwrap();
    assert!(data.get_num_samples() == 6);
    assert!(data.get_num_bytes() == 12);
}

#[test]
fn samples_can_be_appended_to_f32() {
    let samples_1 = Box::new(WavDataType::FLOAT32(vec![1.0, 2.0, 3.0]));
    let samples_2 = Box::new(WavDataType::FLOAT32(vec![4.0, 5.0, 6.0]));
    let mut data = WavData::new_from_data(samples_1);
    data.append_samples(samples_2).unwrap();
    assert!(data.get_num_samples() == 6);
    assert!(data.get_num_bytes() == 24);
}

#[test]
fn samples_can_be_appended_to_f64() {
    let samples_1 = Box::new(WavDataType::FLOAT64(vec![1.0, 2.0, 3.0]));
    let samples_2 = Box::new(WavDataType::FLOAT64(vec![4.0, 5.0, 6.0]));
    let mut data = WavData::new_from_data(samples_1);
    data.append_samples(samples_2).unwrap();
    assert!(data.get_num_samples() == 6);
    assert!(data.get_num_bytes() == 48);
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
fn chunk_can_be_added_and_queried() {
    let cid_test: u32 = 0x54534554;
    let samples = Box::new(WavDataType::PCM8(vec![1, 2, 3]));
    let mut wav_data = WavData::new_from_data(samples);
    let chunk_data = Box::new(vec![0x05, 0x06, 0x07, 0x08]);
    let chunk = Chunk::new_from_data(cid_test, chunk_data);
    wav_data.add_chunk(chunk);

    let chunk_list = wav_data.get_chunks();
    assert!(chunk_list[0].chunk_id == cid_test);
}
