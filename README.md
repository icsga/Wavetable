# Wavetable

This is a library for generating and using wavetables. It is currently used in
the [Yazz software synthesizer](https://github.com/icsga/yazz).

The library (will) offer(s) support for importing wavetables from wave files,
bandlimiting the tables to avoid aliasing, storing tables in a compressed
format and getting samples from the tables.

While functional, this is not yet in a very useful state. The code is currently
being reworked into a library and needs a lot of refactoring, testing and
optimization. There is also no documentation yet, and some features are still
missing.

## TODO

- Support other frequency ranges than single octaves for the bandlimited tables.
- Add FFT analysis of imported waves to create bandlimited tables.
- Add support for storing tables as list of harmonics instead of samples.
- Add support for more wave file formats (currently only floats).
- Add tests.
- Add more examples.
- Add documentation.
