//! Manages access to wavetables.
//!
//! Has a cache with wavetables, handing references out to clients asking for
//! a table.

use super::Float;
use super::{Wavetable, WavetableRef};
use super::WtCreator;
use super::WtReader;

use log::{info, trace, warn};
use serde::{Serialize, Deserialize};

use std::collections::HashMap;
use std::sync::Arc;

const NUM_PWM_TABLES: usize = 64;

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
    /// The wavetable added will contain waves for sine, triangle, saw and
    /// square, 2048 samples per wave, bandlimited with one table per octave,
    /// for 11 octaves, covering the full range of MIDI notes for standard
    /// tuning.
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
    /// wt_manager.load_table(&mut wt_info, fallback);
    /// ```
    pub fn load_table(&mut self, wt_info: &mut WtInfo, fallback: WavetableRef) {
        let result = self.reader.read_file(&wt_info.filename);
        let table = if let Ok(wt) = result {
            wt_info.valid = true;
            wt
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

