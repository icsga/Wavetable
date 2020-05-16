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

use log::{info, trace, warn};
use serde::{Serialize, Deserialize};

use std::collections::HashMap;
use std::sync::Arc;

const NUM_PWM_TABLES: usize = 64;

/// Identifies a wavetable.
///
/// The ID is an internal identifier used for referencing the table.
/// The valid flag is set to true if the wavetable was initialized
/// successfully, false otherwise.
/// The name is a string that can be displayed to the user. It is usually the
/// filename without the path.
/// The filename is the full patch to the wave file
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WtInfo {
    pub id: usize,       // ID of wavetable, used as reference
    pub valid: bool,     // True if wavetable file exists
    pub name: String,    // Name of the wavetable
    pub filename: String // Wavetable filename, empty if internal table
}

pub struct WtManager {
    sample_rate: Float,
    cache: HashMap<usize, WavetableRef>,
    reader: WtReader,
}

impl WtManager {
    /// Generate a new WtManager instance.
    ///
    /// The data_dir is the name of the directory that is used for loading
    /// wavetable files.
    ///
    /// ```
    /// use wavetable::WtManager;
    ///
    /// let wt_manager = WtManager::new(44100.0, "data");
    /// ```
    pub fn new(sample_rate: Float, data_dir: &str) -> WtManager {
        let cache = HashMap::new();
        let reader = WtReader::new(data_dir);
        WtManager{sample_rate, cache, reader}
    }

    /// Add table containing basic waveshapes with the given ID.
    ///
    /// The wavetable added will contain data for sine, triangle, saw and
    /// square waves, 2048 samples per wave, bandlimited with one table per
    /// octave, for 11 octaves, covering the full range of MIDI notes for
    /// standard tuning.
    ///
    /// The wave indices to use for querying sine, triangle, saw and square
    /// are 0.0, 1/3, 2/3 and 1.0 respectively.
    ///
    /// ```
    /// use wavetable::WtManager;
    ///
    /// let mut wt_manager = WtManager::new(44100.0, "data");
    /// wt_manager.add_basic_tables(0);
    /// ```
    pub fn add_basic_tables(&mut self, id: usize) {
        self.add_to_cache(id, WtCreator::create_default_waves(self.sample_rate));
    }


    /// Add table containing pulse width modulated square waves with the given ID.
    ///
    /// The wavetable added will contain the specified number of square waves
    /// with different amounts of pulse width modulation, 2048 samples per
    /// wave, bandlimited with one table per octave, for 11 octaves, covering
    /// the full range of MIDI notes for standard tuning.
    ///
    /// ```
    /// use wavetable::WtManager;
    ///
    /// let mut wt_manager = WtManager::new(44100.0, "data");
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
    /// let mut wt_manager = WtManager::new(44100.0, "data");
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
    /// consume 11 times more memory than the un-bandlimited version.
    ///
    /// ```
    /// use wavetable::{WtManager, WtInfo};
    ///
    /// let mut wt_manager = WtManager::new(44100.0, "data");
    /// wt_manager.add_basic_tables(0);
    /// let mut wt_info = WtInfo{
    ///         id: 1,
    ///         valid: false,
    ///         name: "TestMe".to_string(),
    ///         filename: "TestMe.wav".to_string()};
    /// let fallback = if let Some(table) = wt_manager.get_table(0) {
    ///     table
    /// } else {
    ///     panic!();
    /// };
    /// wt_manager.load_table(&mut wt_info, fallback, false);
    /// ```
    pub fn load_table(&mut self, wt_info: &mut WtInfo, fallback: WavetableRef, bandlimit: bool) {
        let result = self.reader.read_file(&wt_info.filename);
        let table = if let Ok(wt) = result {
            wt_info.valid = true;
            if bandlimit {
                let harmonics = wt.convert_to_harmonics(1024);
                let mut wt_bandlimited = Wavetable::new(wt.table.len(), 11, 2048);
                wt_bandlimited.insert_harmonics(&harmonics, self.sample_rate).unwrap();
                Arc::new(wt_bandlimited)
            } else {
                wt
            }
        } else {
            wt_info.valid = false;
            fallback.clone()
        };
        self.add_to_cache(wt_info.id, table);
    }

    // Adds a wavetable with the given ID to the internal cache.
    fn add_to_cache(&mut self, id: usize, wt: WavetableRef) {
        self.cache.insert(id, wt);
    }
}

