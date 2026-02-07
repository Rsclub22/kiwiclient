[updated 11 July 2025]

# KiwiClient

This is the version v1.6 Python client for KiwiSDR. It allows you to:

* Receive data streams with audio samples, IQ samples, S-meter and waterfall data
* Issue commands to the KiwiSDR

## Install

Visit [github.com/jks-prv/kiwiclient](https://github.com/jks-prv/kiwiclient)  
Click the green `'Code'` button and select `'Download ZIP'`  
You should find the file `'kiwiclient-master.zip'` in your download directory.  
Copy it to the appropriate destination and unzip it if necessary.  
Perhaps rename the resulting directory `'kiwiclient-master'` to just `'kiwiclient'`.  
Change to that directory.  
Here you will find a `'Makefile'` containing various usage examples.  
Assuming your system has `'Make'` and `'Python3'` installed type `'make help'` to get started.  
Or without `'Make'` type `'python3 kiwirecorder.py --help'`  
It is also possible to use the `'git'` tools to checkout a kiwiclient clone that is easier to keep updated.  
For example `'git clone https://github.com/jks-prv/kiwiclient.git'`  

## Dependencies

Python3 is required.

Make sure the Python package `'numpy'` is installed.  
On many Linux distributions the command would be similar to `'apt install python3-numpy'`  
On macOS try `'pip3 install numpy'` or perhaps `'python3 -m pip install numpy'`

Same for package `'chunkmuncher'` if using Python3.13 or later (as module `'chunk'` was removed in 3.13)

## Resampling

If you want high-quality resampling based on libsamplerate (SRC) you should build the version
included with KiwiClient that has fixes rather than using the standard python-samplerate package.  
Follow these steps. Ask on the Kiwi forum if you have problems: [forum.kiwisdr.com](https://forum.kiwisdr.com)
* Install the Python package `'cffi'`
* Install the `'libsamplerate'` library using your system's package manager.
Note: this is not the Python package `'samplerate'` but the native code library `'libsamplerate'`
(e.g. x86\_64 or arm64).
    * Windows: download from `'github.com/libsndfile/libsamplerate/releases'`
    * Linux: use a package manager, e.g. `'apt install libsamplerate'` or `'libsamplerate0'`
    * macOS: use a package manager like brew: `'brew install libsamplerate'`
* Run the samplerate module builder `'make samplerate_build'`.
This generates a Python wrapper around `'libsamplerate'` in the file `'samplerate/_src.py'`
* Install `'pytest'` using the Python package manager or perhaps `'pip3 install pytest'`
* Test by running `'make samplerate_test'`
* If your system says `'pytest'` isn't found try `'make samplerate_test2'`

If you can't build the Kiwi version then install the regular Python package: `'pip3 install samplerate'`  
If either samplerate module is not found then low-quality resampling based on linear interpolation is used.  
A message indicating which resampler is being used will be shown.  
Set the environment variable `'USE_LIBSAMPLERATE'` to '`False'` to force the linear interpolator to be used.

## Demo code

The following demo programs are provided. Use the `--help` argument to see all program options.

* `kiwirecorder`: Record audio to WAV files, with squelch. Option `--wf` prints various waterfall statistics. <br> Adding option `--wf-png` records the waterfall as a PNG file. `--help` for more info.
* `kiwiwfrecorder`: Specialty program. Saves waterfall data and GPS timestamps to .npy format file.
* `kiwifax`: Decode radiofax and save as PNGs, with auto start, stop, and phasing.
* `kiwiclientd`: Plays Kiwi audio on sound cards (real & virtual) for use by programs like fldigi and wsjtx.
    Implements hamlib rigctl network interface so the Kiwi freq & mode can be controlled by these programs. The
    new option `--http-prefix` lets `kiwiclientd` connect through a reverse proxy by specifying the HTTP path
    prefix used by the proxy (e.g. `/kiwi`).
    Option `--https` enables TLS connections when the Kiwi server is only
    reachable via HTTPS.
* `kiwi_nc`: Deprecated. Use the `--nc` option with `kiwirecorder`. Command line pipeline tool in the style of `netcat`.
Example: streaming IQ samples to `dumphfdl` (see the `Makefile` target `dumphfdl`).

The `Makefile` contains numerous examples of how to use these programs.

## IS0KYB micro tools

Two utilities have been added to simplify the waterfall data acquisition/storage and data analysis.
The SNR ratio (a la Pierre Ynard) is computed each time.
There is now the possibility to change zoom level and offset frequency.

* `microkiwi_waterfall.py`: launch this program with no filename and just the SNR will be computed, with a filename, the raw waterfall data is saved. Launch with `--help` to list all options.
* `waterfall_data_analysis.ipynb`: this is a demo jupyter notebook to interactively analyze waterfall data. Easily transformable into a standalone python program.

The data is, at the moment, transferred in uncompressed format.

## TCP Socket Optimizations for Reduced Latency

All client programs now include TCP socket optimizations to reduce latency and prevent excessive buffering:

* **TCP_NODELAY** (enabled by default): Disables Nagle's algorithm to send packets immediately, reducing latency
* **TCP Keepalive** (enabled by default): Detects stale connections with configurable timing parameters
* **Receive Buffer Limiting**: Optional socket receive buffer size limit to prevent buffering large amounts of data during connection issues

### Options:
* `--tcp-nodelay` / `--no-tcp-nodelay`: Enable/disable TCP_NODELAY (default: enabled)
* `--socket-rcvbuf <bytes>`: Set socket receive buffer size (e.g., `--socket-rcvbuf 65536` for 64KB)
* `--tcp-keepalive` / `--no-tcp-keepalive`: Enable/disable TCP keepalive (default: enabled)
* `--tcp-keepidle <seconds>`: TCP keepalive idle time (Linux only, default: 10)
* `--tcp-keepintvl <seconds>`: TCP keepalive interval (Linux only, default: 5)
* `--tcp-keepcnt <count>`: TCP keepalive probe count (Linux only, default: 3)

### Choosing a Receive Buffer Size:

The `--socket-rcvbuf` option limits how much data the OS will buffer. This is important to prevent accumulating excessive data (20+ minutes) during connection issues.

**How to calculate an appropriate buffer size:**
* Audio bitrate depends on sample rate and compression
* Typical KiwiSDR audio: ~12 kHz sample rate × 16 bits × compression ratio ≈ 24-96 kbps
* Buffer size = (desired max latency in seconds) × (bitrate in bytes/sec)

**Recommended buffer sizes:**
* **65536 bytes (64 KB)**: Good for ~5-10 seconds of buffering (recommended for most users)
* **131072 bytes (128 KB)**: Moderate buffering, ~10-20 seconds
* **32768 bytes (32 KB)**: Minimal buffering, ~2-5 seconds (for very low latency)
* **No limit (default)**: May buffer several MB (minutes to hours of audio)

Example: For 12 kHz audio at 8 kbps compressed, 64KB buffer = ~64 seconds. For uncompressed at 192 kbps, 64KB = ~2.5 seconds.

### Example usage:
```bash
# Reduce latency with TCP_NODELAY and limit buffer to 64KB
python3 kiwirecorder.py -s kiwisdr.example.com -p 8073 -f 7200 -m usb --socket-rcvbuf 65536

# Disable TCP_NODELAY if needed
python3 kiwirecorder.py -s kiwisdr.example.com -p 8073 -f 7200 -m usb --no-tcp-nodelay

# Configure aggressive keepalive settings
python3 kiwiclientd.py -s kiwisdr.example.com -f 14074 -m usb --tcp-keepidle 5 --tcp-keepintvl 3
```

These optimizations help prevent the client from buffering excessive amounts of data (e.g., 20+ minutes) during connection problems and reduce overall latency in the audio stream.

## Guide to the code

### kiwirecorder.py
* Can record audio data, IQ samples, and waterfall data.
* The complete list of options can be obtained by `python3 kiwirecorder.py --help` or `make help`.
* It is possible to record from more than one KiwiSDR simultaneously, see again `--help`.
* For recording IQ samples there is the `-w` or `--kiwi-wav` option: this writes a .wav file  
which includes GNSS timestamps (see below).
* The `--netcat` option can stream raw or .wav-formatted samples to standard output.
* Kiwirecorder can "camp" onto an existing KiwiSDR audio channel. See the `--camp-chan` option.
* AGC options can be specified in a YAML-formatted file, `--agc-yaml` option, see `default_agc.yaml`.
* Scanning options (with optional squelch) can be specified in a YAML-formatted file, `--scan-yaml` option.  
See the file `SCANNING` for detailed info.
* Note the above YAML options need PyYAML to be installed.
* See the `Makefile` for many usage examples.
* Ask on the KiwiSDR Forum for help: [forum.kiwisdr.com](https://forum.kiwisdr.com)

### kiwiclient.py

Base class for receiving websocket data from a KiwiSDR.
It provides the following methods which can be used in derived classes:

* `_process_audio_samples(self, seq, samples, rssi)`: audio samples
* `_process_iq_samples(self, seq, samples, rssi, gps)`: IQ samples
* `_process_waterfall_samples(self, seq, samples)`: waterfall data

## IQ .wav files with GNSS timestamps
### kiwirecorder.py configuration
* Use the option `-m iq --kiwi-wav --station=[name]` for recording IQ samples with GNSS time stamps.
* The resulting .wav files contains non-standard WAV chunks with GNSS timestamps.
* If a directory with name `gnss_pos/` exists, a text file `gnss_pos/[name].txt` will be created which contains latitude and longitude as provided by the KiwiSDR; existing files are overwritten.

### Working with the recorded .wav files
* There is an octave extension for reading such WAV files, see `read_kiwi_wav.cc` where the details of the non-standard WAV chunk can be found; it needs to be compiled in this way: `mkoctfile read_kiwi_wav.cc`.
* For using read_kiwi_wav an octave function `proc_kiwi_iq_wav.m` is provided; type `help proc_kiwi_iq_wav` in octave for documentation.

[end-of-document]
