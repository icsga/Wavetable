//! Manages access to wavetables.
//!
//! The wavetable manager is a central place for managing wavetable
//! instances. It holds wavetables in a cache, handing references to them to
//! clients, thereby avoiding the need for multiple instances of a single
//! table.

use super::Float;
use super::{Wavetable, WavetableRef};
use super::WtCreator;
use super::WtReader;

use serde::{Serialize, Deserialize};

use std::collections::HashMap;
use std::sync::Arc;

/// Identifies a wavetable.
///
/// The ID is an internal identifier used for referencing the table.
/// The valid flag is set to true if the wavetable was initialized
/// successfully, false otherwise.
/// The name is a string that can be displayed to the user. It is usually the
/// filename without the path.
/// The filename is the full path to the wave file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WtInfo {
    pub id: usize,       // ID of wavetable, used as reference
    pub valid: bool,     // True if wavetable file exists
    pub name: String,    // Name of the wavetable
    pub filename: String // Wavetable filename, empty if internal table
}

pub struct WtManager {
    sample_rate: Float,
    num_octaves: usize,
    cache: HashMap<usize, WavetableRef>,
    reader: WtReader,
}

impl WtManager {
    /// Generate a new WtManager instance.
    ///
    /// sample_rate is required to correctly calculate the number of harmonics
    /// to generate for bandlimited octave tables, to avoid aliasing.
    ///  
    /// ```
    /// use wavetable::WtManager;
    ///
    /// let wt_manager = WtManager::new(44100.0);
    /// ```
    pub fn new(sample_rate: Float) -> WtManager {
        let cache = HashMap::new();
        let reader = WtReader::new("");
        WtManager{sample_rate, num_octaves: 11, cache, reader}
    }

    /// Set the number of bandlimited tables to create.
    ///
    /// num_octaves determines how many of the bandlimited tables to generate.
    /// The default is 11, covering the full range of MIDI notes for standard
    /// tuning. Setting it to a lower value can save some memory.
    ///
    /// Setting it to a higher value higher than 11 will usually not have much
    /// benefit, since the base frequency of an octave higher than 11 will be
    /// above the Nyquist frequency, so the table will be effectively empty.
    ///
    /// ```
    /// use wavetable::WtManager;
    ///
    /// let mut wt_manager = WtManager::new(44100.0);
    /// wt_manager.set_num_octaces(8);
    /// ```
    pub fn set_num_octaces(&mut self, num_octaves: usize) {
        self.num_octaves = num_octaves;
    }

    /// Add table containing basic waveshapes with the given ID.
    ///
    /// The wavetable added will contain data for sine, triangle, saw and
    /// square waves, 2048 samples per wave, bandlimited with one table per
    /// octave.
    ///
    /// The wave indices to use for querying sine, triangle, saw and square
    /// are 0.0, 1/3, 2/3 and 1.0 respectively.
    ///
    /// ```
    /// use wavetable::WtManager;
    ///
    /// let mut wt_manager = WtManager::new(44100.0);
    /// wt_manager.add_basic_tables(0);
    /// ```
    pub fn add_basic_tables(&mut self, id: usize) {
        self.add_to_cache(id, WtCreator::create_default_waves(self.sample_rate));
    }


    /// Add table containing pulse width modulated square waves with the given ID.
    ///
    /// The wavetable added will contain the specified number of square waves
    /// with different amounts of pulse width modulation, 2048 samples per
    /// wave, bandlimited with one table per octave.
    ///
    /// ```
    /// use wavetable::WtManager;
    ///
    /// let mut wt_manager = WtManager::new(44100.0);
    /// wt_manager.add_pwm_tables(1, 64);
    /// ```
    pub fn add_pwm_tables(&mut self, id: usize, num_pwm_tables: usize) {
        self.add_to_cache(id, WtCreator::create_pwm_waves(self.sample_rate, num_pwm_tables));
    }

    /// Get a single wavetable by id from the cache.
    ///
    /// ```
    /// use wavetable::WtManager;
    ///
    /// let mut wt_manager = WtManager::new(44100.0);
    /// wt_manager.add_basic_tables(0);
    /// let table_ref = wt_manager.get_table(0);
    /// ```
    pub fn get_table(&self, id: usize) -> Option<WavetableRef> {
        if self.cache.contains_key(&id) {
            Some(self.cache.get(&id).unwrap().clone())
        } else {
            None
        }
    }

    /// Set the path for loading and saving wave files.
    ///
    /// ```
    /// use wavetable::{WtManager, WtInfo};
    ///
    /// let mut wt_manager = WtManager::new(44100.0);
    /// wt_manager.set_path("data");
    /// ```
    pub fn set_path(&mut self, path: &str) {
        self.reader.set_path(path);
    }

    /// Loads a wavetable file and adds the table to the cache.
    ///
    /// Tries to load the table from the given file and put it into the cache.
    /// If loading the file fails, the provided fallback table is inserted
    /// instead.
    ///
    /// The WtInfo::valid flag is set to true if loading was successfull, false
    /// if it failed.
    ///
    /// If the flag bandlimit is set to true, the table will automatically be
    /// converted to a bandlimited version, consisting of 11 tables per wav
    /// shape with reduced number of harmonics, which means the data will
    /// consume 11 times more memory than the non-bandlimited version.
    ///
    /// The number of octave tables to create can be configured with
    /// set_num_octaves().
    ///
    /// ```
    /// use wavetable::{WtManager, WtInfo};
    ///
    /// let mut wt_manager = WtManager::new(44100.0);
    /// wt_manager.add_basic_tables(0);
    /// let mut wt_info = WtInfo{
    ///         id: 1,
    ///         valid: false,
    ///         name: "TestMe".to_string(),
    ///         filename: "TestMe.wav".to_string()};
    /// let fallback = wt_manager.get_table(0).unwrap();
    /// wt_manager.load_table(&mut wt_info, fallback, false);
    /// ```
    pub fn load_table(&mut self, wt_info: &mut WtInfo, fallback: WavetableRef, bandlimit: bool) {
        let result = self.reader.read_file(&wt_info.filename, Some(2048));
        let table = if let Ok(wt) = result {
            wt_info.valid = true;
            if bandlimit {
                WtManager::bandlimit(wt, self.num_octaves, self.sample_rate)
            } else {
                wt
            }
        } else {
            wt_info.valid = false;
            fallback.clone()
        };
        self.add_to_cache(wt_info.id, table);
    }

    pub fn bandlimit(wt: WavetableRef, num_octaves: usize, sample_rate: Float) -> WavetableRef {
        let spectrum = wt.get_freq_spectrum();
        let mut wt_bandlimited = Wavetable::new(wt.table.len(), num_octaves, wt.num_samples);
        wt_bandlimited.add_frequencies(&spectrum, sample_rate).unwrap();
        Arc::new(wt_bandlimited)
    }

    pub fn write_table(&self, wt_ref: WavetableRef, filename: &str) {
        self.reader.write_file(wt_ref, filename).unwrap();
    }

    // Adds a wavetable with the given ID to the internal cache.
    pub fn add_to_cache(&mut self, id: usize, wt: WavetableRef) {
        self.cache.insert(id, wt);
    }
}

