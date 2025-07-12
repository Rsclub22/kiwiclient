#!/usr/bin/env python
## -*- python -*-
#
# Streams sound from a kiwisdr channel to a (virtual or not) sound card,
# allowing the user to process kiwisdr signals with programs like fldigi,
# wsjtx, etc.
# Provides a hamlib rictld backend to change frequency and modulation of
# the kiwisdr channel.
#
# Uses the SoundCard python module, which can stream sound to
# coreaudio (MacOS), mediafoundation (Windows), and pulseaudio (Linux)

import array
import logging
import os
import struct
import sys
import time
import copy
import threading
import gc
import math
from flask import Flask, jsonify
from werkzeug.serving import make_server
import soundcard as sc
import numpy as np
from copy import copy
from traceback import print_exc
from kiwi import KiwiSDRStream, KiwiWorker
from optparse import OptionParser
from optparse import OptionGroup
import socket
import re
import json

# global status dictionary used by the web interface
status_data = {
    'connected': False,
    'status': 'disconnected',
    'server': '',
    'mode': '',
    'frequency': 0,
    'station': ''
}


app = Flask(__name__)

@app.route('/')
def status_page():
    html = (
        '<html><head><meta charset="utf-8"><title>KiwiClient Status</title></head>'
        '<body>'
        '<h1>KiwiClient Status</h1>'
        f'<p>Verbunden: {"Ja" if status_data["connected"] else "Nein"}</p>'
        f'<p>Status: {status_data["status"]}</p>'
        f'<p>Server: {status_data["server"]}</p>'
        f'<p>Einstellung: Mode={status_data["mode"]} '
        f'Freq={status_data["frequency"]} Station={status_data["station"]}</p>'
        '</body></html>'
    )
    return html


@app.route('/json')
def status_json():
    return jsonify(status_data)

HAS_RESAMPLER = True
try:
    ## if available use libsamplerate for resampling
    from samplerate import Resampler
except ImportError:
    ## otherwise linear interpolation is used
    HAS_RESAMPLER = False

class KiwiSoundRecorder(KiwiSDRStream):
    def __init__(self, options):
        super(KiwiSoundRecorder, self).__init__()
        self._options = options
        self._type = 'SND'
        freq = options.frequency
        options.S_meter = -1
        options.stats = False
        #logging.info("%s:%s freq=%d" % (options.server_host, options.server_port, freq))
        self._freq = freq
        self._ifreq = options.ifreq
        self._modulation = self._options.modulation
        self._lowcut = self._options.lp_cut
        self._highcut = self._options.hp_cut
        self._start_ts = None
        self._start_time = None
        self._squelch = Squelch(self._options) if options.thresh is not None else None
        self._last_gps = dict(zip(['last_gps_solution', 'dummy', 'gpssec', 'gpsnsec'], [0,0,0,0]))
        self._resampler = None
        self._output_sample_rate = 0

    def _init_player(self):
        if hasattr(self, 'player'):
            self._player.__exit__(exc_type=None, exc_value=None, traceback=None)
        options = self._options
        speaker = sc.get_speaker(options.sounddevice)
        rate = self._output_sample_rate
        if speaker is None:
            if options.sounddevice is None:
                print('Using default sound device. Specify --sound-device?')
                options.sounddevice = 'default'
            else:
                print("Could not find %s, using default", options.sounddevice)
                speaker = sc.default_speaker()

        # pulseaudio has sporadic failures, retry a few times
        for i in range(0,10):
            try:
                self._player = speaker.player(samplerate=rate)
                self._player.__enter__()
                break
            except Exception as ex:
                print("speaker.player failed with ", ex)
                time.sleep(0.1)
                pass

    def _setup_rx_params(self):
        self.set_name(self._options.user)
        lowcut = self._lowcut
        if self._modulation == 'am':
            # For AM, ignore the low pass filter cutoff
            lowcut = -self._highcut if lowcut is not None else lowcut
        self.set_mod(self._modulation, lowcut, self._highcut, self._freq)
        if self._options.agc_gain != None:
            self.set_agc(on=False, gain=self._options.agc_gain)
        else:
            self.set_agc(on=True)
        if self._options.compression is False:
            self._set_snd_comp(False)
        if self._options.nb is True:
            gate = self._options.nb_gate
            if gate < 100 or gate > 5000:
                gate = 100
            thresh = self._options.nb_thresh
            if thresh < 0 or thresh > 100:
                thresh = 50
            self.set_noise_blanker(gate, thresh)
        if self._options.de_emp is True:
            self.set_de_emp(1)
        self._output_sample_rate = int(self._sample_rate)
        if self._options.resample > 0:
            self._output_sample_rate = self._options.resample
            self._ratio = float(self._output_sample_rate)/self._sample_rate
            logging.info('resampling from %g to %d Hz (ratio=%f)' % (self._sample_rate, self._options.resample, self._ratio))
            if not HAS_RESAMPLER:
                logging.info("libsamplerate not available: linear interpolation is used for low-quality resampling. "
                             "(pip/pip3 install samplerate)")
        if self._ifreq is not None:
            if self._modulation != 'iq':
                logging.warning('Option --if %.1f only valid for IQ modulation, ignored' % self._ifreq)
            elif self._output_sample_rate < self._ifreq * 4:
                logging.warning('Sample rate %.1f is not enough for --if %.1f, ignored. Use --resample %.1f' % (
                    self._output_sample_rate, self._ifreq, self._ifreq * 4))
        self._init_player()

    def _process_audio_samples(self, seq, samples, rssi, fmt):
        if self._options.quiet is False:
            sys.stdout.write('\rBlock: %08x, RSSI: %6.1f' % (seq, rssi))
            sys.stdout.flush()

        if self._options.resample > 0:
            if HAS_RESAMPLER:
                ## libsamplerate resampling
                if self._resampler is None:
                    self._resampler = Resampler(converter_type='sinc_best')
                samples = np.round(self._resampler.process(samples, ratio=self._ratio)).astype(np.int16)
            else:
                ## resampling by linear interpolation
                n  = len(samples)
                xa = np.arange(round(n*self._ratio))/self._ratio
                xp = np.arange(n)
                samples = np.round(np.interp(xa,xp,samples)).astype(np.int16)


        # Convert the int16 samples [-32768,32,767] to the floating point
        # samples [-1.0,1.0] SoundCard expects
        fsamples = samples.astype(np.float32)
        fsamples /= 32768
        self._player.play(fsamples)

    def _process_stereo_samples_raw(self, seq, data):
        if self._options.quiet is False:
            sys.stdout.write('\rBlock: %08x' % seq)
            sys.stdout.flush()

        n = len(data)//4

        if self._options.resample == 0 or HAS_RESAMPLER:
            ## convert bytes into an array
            s = np.ndarray((n,2), dtype='>h', buffer=data).astype(np.float32) / 32768

        if self._options.resample > 0:
            if HAS_RESAMPLER:
                ## libsamplerate resampling
                if self._resampler is None:
                    self._resampler = Resampler(channels=2, converter_type='sinc_best')
                s = self._resampler.process(s, ratio=self._ratio)
            else:
                ## resampling by linear interpolation
                m  = int(round(n*self._ratio))
                xa = np.arange(m)/self._ratio
                xp = np.arange(n)
                s  = np.ndarray((m,2), dtype=np.float32)
                s[:, 0] = np.interp(xa, xp, data[0::2] / 32768)
                s[:, 1] = np.interp(xa, xp, data[1::2] / 32768)

        if self._ifreq is not None and self._output_sample_rate >= 4 * self._ifreq:
            # view as complex after possible resampling - no copying.
            cs = s.view(dtype=np.complex64)
            l = len(cs)
            # get final phase value
            stopph = self.startph + 2 * np.pi * l * self._ifreq / self._output_sample_rate
            # all the steps needed
            steps = 1j*np.linspace(self.startph, stopph, l, endpoint=False, dtype=np.float32)
            # shift frequency and get back to a 2D array
            s = (cs * np.exp(steps)[:, None]).view(np.float32)
            # save phase  for next time, modulo 2π
            self.startph = stopph % (2*np.pi)

        self._player.play(s)

    # phase for frequency shift
    startph = np.float32(0)

    def _process_iq_samples(self, seq, samples, rssi, gps, fmt):
        if self._options.quiet is False:
            sys.stdout.write('\rBlock: %08x, RSSI: %6.1f' % (seq, rssi))
            sys.stdout.flush()

        if self._squelch:
            is_open = self._squelch.process(seq, rssi)
            if not is_open:
                self._start_ts = None
                self._start_time = None
                return

        ##print gps['gpsnsec']-self._last_gps['gpsnsec']
        self._last_gps = gps

        if self._options.resample == 0 or HAS_RESAMPLER:
            ## convert list of complex numbers into an array
            s = np.ndarray((len(samples),2), dtype=np.float32)
            s[:, 0] = np.real(samples).astype(np.float32) / 32768
            s[:, 1] = np.imag(samples).astype(np.float32) / 32768

        if self._options.resample > 0:
            if HAS_RESAMPLER:
                ## libsamplerate resampling
                if self._resampler is None:
                    self._resampler = Resampler(channels=2, converter_type='sinc_best')
                s = self._resampler.process(s, ratio=self._ratio)
            else:
                ## resampling by linear interpolation
                n  = len(samples)
                m  = int(round(n*self._ratio))
                xa = np.arange(m)/self._ratio
                xp = np.arange(n)
                s  = np.ndarray((m,2), dtype=np.float32)
                s[:, 0] = np.interp(xa, xp, np.real(samples).astype(np.float32) / 32768)
                s[:, 1] = np.interp(xa, xp, np.imag(samples).astype(np.float32) / 32768)


        if self._ifreq is not None and self._output_sample_rate >= 4 * self._ifreq:
            # view as complex after possible resampling - no copying.
            cs = s.view(dtype=np.complex64)
            l = len(cs)
            # get final phase value
            stopph = self.startph + 2 * np.pi * l * self._ifreq / self._output_sample_rate
            # all the steps needed
            steps = 1j*np.linspace(self.startph, stopph, l, endpoint=False, dtype=np.float32)
            # shift frequency and get back to a 2D array
            s = (cs * np.exp(steps)[:, None]).view(np.float32)
            # save phase  for next time, modulo 2π
            self.startph = stopph % (2*np.pi)

        self._player.play(s)

        # no GPS or no recent GPS solution
        last = gps['last_gps_solution']
        if last == 255 or last == 254:
            self._options.status = 3

    def _on_sample_rate_change(self):
        if self._options.resample == 0:
            # if self._output_sample_rate == int(self._sample_rate):
            #    return
            # reinitialize player if the playback sample rate changed
            self._output_sample_rate = int(self._sample_rate)
            self._init_player()


class WebKiwiWorker(KiwiWorker):
    """Worker that updates the status dictionary for the web interface."""

    def run(self):
        status_data['server'] = f"{self._options.server_host}:{self._options.server_port}"
        status_data['frequency'] = self._options.frequency
        status_data['mode'] = self._options.modulation
        status_data['station'] = self._options.station or ''

        self.connect_count = self._options.connect_retries
        self.busy_count = self._options.busy_retries
        if self._delay_run:
            time.sleep(3)

        while self._do_run():
            status_data['status'] = 'connecting'
            try:
                self._recorder.connect(self._options.server_host, self._options.server_port)
            except Exception as e:
                logging.warn("Failed to connect, sleeping and reconnecting error='%s'" % e)
                status_data['connected'] = False
                status_data['status'] = 'timeout'
                self.connect_count -= 1
                if self._options.connect_retries > 0 and self.connect_count == 0:
                    break
                if self._options.connect_timeout > 0:
                    self._event.wait(timeout=self._options.connect_timeout)
                continue

            try:
                self._recorder.open()
                status_data['connected'] = True
                status_data['status'] = 'running'
                while self._do_run():
                    self._recorder.run()
                    if self._rigctld:
                        self._rigctld.run()

            except KiwiServerTerminatedConnection as e:
                status_data['status'] = 'timeout'
                logging.info("%s:%s %s. Reconnecting after 5 seconds" % (
                    self._options.server_host, self._options.server_port, e))
                self._recorder.close()
                if self._options.no_api:
                    break
                self._recorder._start_ts = None
                self._event.wait(timeout=5)
                continue

            except KiwiTooBusyError:
                status_data['status'] = 'too busy'
                self.busy_count -= 1
                if self._options.busy_retries > 0 and self.busy_count == 0:
                    break
                logging.warn("%s:%d Too busy now. Reconnecting after %d seconds" % (
                    self._options.server_host, self._options.server_port, self._options.busy_timeout))
                if self._options.busy_timeout > 0:
                    self._event.wait(timeout=self._options.busy_timeout)
                continue

            except KiwiRedirectError as e:
                prev = self._options.server_host + ':' + str(self._options.server_port)
                uri = str(e).split(':')
                self._options.server_host = uri[1][2:]
                self._options.server_port = uri[2]
                status_data['server'] = f"{self._options.server_host}:{self._options.server_port}"
                status_data['status'] = 'redirect'
                logging.warn("%s Too busy now. Redirecting to %s:%s" % (
                    prev, self._options.server_host, self._options.server_port))
                self._event.wait(timeout=2)
                continue

            except KiwiTimeLimitError:
                status_data['status'] = 'time limit'
                break

            except Exception:
                status_data['status'] = 'error'
                print_exc()
                break

        status_data['connected'] = False
        self._run_event.clear()
        self._recorder.close()
        self._recorder._close_func()
        if self._rigctld:
            self._rigctld.close()

def options_cross_product(options):
    """build a list of options according to the number of servers specified"""
    def _sel_entry(i, l):
        """if l is a list, return the element with index i, else return l"""
        return l[min(i, len(l)-1)] if type(l) == list else l

    l = []
    multiple_connections = 0
    for i,s in enumerate(options.rigctl_port):
        opt_single = copy(options)
        opt_single.rigctl_port = s
        opt_single.status = 0

        # time() returns seconds, so add pid and host index to make timestamp unique per connection
        opt_single.ws_timestamp = int(time.time() + os.getpid() + i) & 0xffffffff
        for x in ['server_host', 'server_port', 'password', 'tlimit_password',
                  'frequency', 'agc_gain', 'station', 'user', 'sounddevice',
                  'rigctl_port', 'http_prefix']:
            opt_single.__dict__[x] = _sel_entry(i, opt_single.__dict__[x])
        l.append(opt_single)
        multiple_connections = i
    return multiple_connections,l

def get_comma_separated_args(option, opt, value, parser, fn):
    values = [fn(v.strip()) for v in value.split(',')]
    setattr(parser.values, option.dest, values)
##    setattr(parser.values, option.dest, map(fn, value.split(',')))

def join_threads(snd):
    [r._event.set() for r in snd]
    [t.join() for t in threading.enumerate() if t is not threading.current_thread()]


def start_web_server(port):
    """Start the status web server on the given port using Flask."""
    server = make_server('0.0.0.0', port, app)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

def udp_status_listener(_kiwi_recorder, udp_port, station_filter=None):
    """Listen for Kiwi STATUS messages on UDP and update the status dict."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", udp_port))
    print(f"[UDP] Listening for STATUS on UDP port {udp_port} ...")
    while True:
        data, addr = sock.recvfrom(4096)
        message = data.decode(errors='ignore')
        if message.startswith("STATUS"):
            parts = re.findall(r'"[^"]*"|\S+', message)
            if len(parts) >= 8:
                stn = parts[1].strip('"')
                val2 = parts[5]
                freq = parts[7]
                freq_float = int(freq) / 10.0
                # Determine mode from value
                if val2 == "1":
                    mode = "USB" if freq_float > 10000.0 else "LSB"
                else:
                    mode = "CW"
                # Filter nach Station, falls angegeben
                if (station_filter is None) or (stn == station_filter):
                    freq_to_set = freq_float
                    if mode == "CW":
                        freq_to_set -= 0.5  # adjust CW offset
                    _kiwi_recorder.set_mod(mode.lower(), None, None, freq_to_set)
                    _kiwi_recorder._freq = freq_to_set
                    _kiwi_recorder._modulation = mode.lower()
                    status_data['station'] = stn
                    status_data['frequency'] = freq_float
                    status_data['mode'] = mode.lower()

def main():
    # extend the OptionParser so that we can print multiple paragraphs in
    # the help text
    class MyParser(OptionParser):
        def format_description(self, formatter):
            result = []
            for paragraph in self.description:
                result.append(formatter.format_description(paragraph))
            return "\n".join(result[:-1]) # drop last \n

        def format_epilog(self, formatter):
            result = []
            for paragraph in self.epilog:
                result.append(formatter.format_epilog(paragraph))
            return "".join(result)

    usage = "%prog -s SERVER -p PORT -f FREQ -m MODE [other options]"
    description = ["kiwiclientd.py receives audio from a KiwiSDR and plays"
                   " it to a (virtual) sound device. This can be used to"
                   " send KiwiSDR audio to various programs to decode the"
                   " received signals."
                   " This program also accepts hamlib rigctl commands over"
                   " a network socket to change the kiwisdr frequency"
                   " To stream multiple KiwiSDR channels at once, use the"
                   " same syntax, but pass a list of values (where applicable)"
                   " instead of single values. For example, to stream"
                   " two KiwiSDR channels in USB to the virtual sound cards"
                   " kiwisdr0 & kiwisdr1, with the rigctl ports 6400 &"
                   " 6401 respectively, run the following:",
                   "$ kiwiclientd.py -s kiwisdr.example.com -p 8073 -f 10000 -m usb --snddev kiwisnd0,kiwisnd1 --rigctl-port 6400,6401 --enable-rigctl" ,""]
    epilog = [] # text here would go after the options list
    parser = MyParser(usage=usage, description=description, epilog=epilog)
    parser.add_option('-s', '--server-host',
                      dest='server_host', type='string',
                      default='localhost', help='Server host (can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('-p', '--server-port',
                      dest='server_port', type='string',
                      default=8073, help='Server port, default 8073 (can be a comma-separated list)',
                      action='callback',
                      callback_args=(int,),
                      callback=get_comma_separated_args)
    parser.add_option('--http-prefix',
                      dest='http_prefix', type='string', default='',
                      help='HTTP path prefix when using a reverse proxy')
    parser.add_option('--https',
                      dest='https', action='store_true', default=False,
                      help='Use HTTPS (wss) when connecting to the server')
    parser.add_option('--pw', '--password',
                      dest='password', type='string', default='',
                      help='Kiwi login password (if required, can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('--tlimit-pw', '--tlimit-password',
                      dest='tlimit_password', type='string', default='',
                      help='Connect time limit exemption password (if required, can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('-u', '--user',
                      dest='user', type='string', default='kiwiclientd',
                      help='Kiwi connection user name (can be a comma-separated list)',
                      action='callback',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    parser.add_option('--log', '--log-level', '--log_level', type='choice',
                      dest='log_level', default='warn',
                      choices=['debug', 'info', 'warn', 'error', 'critical'],
                      help='Log level: debug|info|warn(default)|error|critical')
    parser.add_option('-q', '--quiet',
                      dest='quiet',
                      default=False,
                      action='store_true',
                      help='Don\'t print progress messages')
    parser.add_option('--tlimit', '--time-limit',
                      dest='tlimit',
                      type='float', default=None,
                      help='Record time limit in seconds. Ignored when --dt-sec used.')
    parser.add_option('--launch-delay', '--launch_delay',
                      dest='launch_delay',
                      type='int', default=0,
                      help='Delay (secs) in launching multiple connections')
    parser.add_option('--connect-retries', '--connect_retries',
                      dest='connect_retries', type='int', default=0,
                      help='Number of retries when connecting to host (retries forever by default)')
    parser.add_option('--connect-timeout', '--connect_timeout',
                      dest='connect_timeout', type='int', default=15,
                      help='Retry timeout(sec) connecting to host')
    parser.add_option('--busy-timeout', '--busy_timeout',
                      dest='busy_timeout',
                      type='int', default=15,
                      help='Retry timeout(sec) when host is busy')
    parser.add_option('--busy-retries', '--busy_retries',
                      dest='busy_retries',
                      type='int', default=0,
                      help='Number of retries when host is busy (retries forever by default)')
    parser.add_option('-k', '--socket-timeout', '--socket_timeout',
                      dest='socket_timeout', type='int', default=10,
                      help='Socket timeout(sec) during data transfers')
    parser.add_option('--restart-sec',
                      dest='restart_sec', type='int', default=10800,
                      help='Restart TCP connection after this many seconds (default 10800)')
    parser.add_option('--OV',
                      dest='ADC_OV',
                      default=False,
                      action='store_true',
                      help='Print "ADC OV" message when Kiwi ADC is overloaded')
    parser.add_option('-v', '-V', '--version',
                      dest='krec_version',
                      default=False,
                      action='store_true',
                      help='Print version number and exit')

    group = OptionGroup(parser, "Audio connection options", "")
    group.add_option('-f', '--freq',
                      dest='frequency',
                      type='string', default=1000,
                      help='Frequency to tune to, in kHz (can be a comma-separated list). '
                        'For sideband modes (lsb/lsn/usb/usn/cw/cwn) this is the carrier frequency. See --pbc option below.',
                      action='callback',
                      callback_args=(float,),
                      callback=get_comma_separated_args)
    group.add_option('--pbc', '--freq-pbc',
                      dest='freq_pbc',
                      action='store_true', default=False,
                      help='For sideband modes (lsb/lsn/usb/usn/cw/cwn) interpret -f/--freq frequency as the passband center frequency.')
    group.add_option('-m', '--modulation',
                      dest='modulation',
                      type='string', default='am',
                      help='Modulation; one of am/amn/amw, sam/sau/sal/sas/qam, lsb/lsn, usb/usn, cw/cwn, nbfm/nnfm, iq (default passband if -L/-H not specified)')
    group.add_option('--ncomp', '--no_compression',
                      dest='compression',
                      default=True,
                      action='store_false',
                      help='Don\'t use audio compression')
    group.add_option('-L', '--lp-cutoff',
                      dest='lp_cut',
                      type='float', default=None,
                      help='Low-pass cutoff frequency, in Hz')
    group.add_option('-H', '--hp-cutoff',
                      dest='hp_cut',
                      type='float', default=None,
                      help='High-pass cutoff frequency, in Hz')
    group.add_option('-r', '--resample',
                      dest='resample',
                      type='int', default=0,
                      help='Resample output file to new sample rate in Hz. The resampling ratio has to be in the range [1/256,256]')
    group.add_option('-T', '--squelch-threshold',
                      dest='thresh',
                      type='float', default=None,
                      help='Squelch threshold, in dB.')
    group.add_option('--squelch-tail',
                      dest='squelch_tail',
                      type='float', default=1,
                      help='Time for which the squelch remains open after the signal is below threshold.')
    group.add_option('-g', '--agc-gain',
                      dest='agc_gain',
                      type='string',
                      default=None,
                      help='AGC gain; if set, AGC is turned off (can be a comma-separated list)',
                      action='callback',
                      callback_args=(float,),
                      callback=get_comma_separated_args)
    group.add_option('--nb',
                      dest='nb',
                      action='store_true', default=False,
                      help='Enable noise blanker with default parameters.')
    group.add_option('--de-emp',
                      dest='de_emp',
                      action='store_true', default=False,
                      help='Enable de-emphasis.')
    group.add_option('--raw',
                      dest='raw',
                      action='store_true', default=False,
                      help='Raw samples processing')
    group.add_option('--if',
                      dest='ifreq',
                      type='float', default=None,
                      help='Intermediate frequency, Hz. Default: no IF')
    group.add_option('--nb-gate',
                      dest='nb_gate',
                      type='int', default=100,
                      help='Noise blanker gate time in usec (100 to 5000, default 100)')
    group.add_option('--nb-th', '--nb-thresh',
                      dest='nb_thresh',
                      type='int', default=50,
                      help='Noise blanker threshold in percent (0 to 100, default 50)')
    parser.add_option_group(group)

    group = OptionGroup(parser, "Sound device options", "")
    group.add_option('--snddev', '--sound-device',
                      dest='sounddevice',
                      type='string', default='',
                      action='callback',
                      help='Sound device to play kiwi audio on (can be comma separated list)',
                      callback_args=(str,),
                      callback=get_comma_separated_args)
    group.add_option('--ls-snd', '--list-sound-devices',
                      dest='list_sound_devices',
                      default=False,
                      action='store_true',
                      help='List available sound devices and exit')
    parser.add_option_group(group)

    group = OptionGroup(parser, "Rig control options", "")
    group.add_option('--rigctl', '--enable-rigctl',
                      dest='rigctl_enabled',
                      default=False,
                      action='store_true',
                      help='Enable rigctld backend for frequency changes.')
    group.add_option('--rigctl-port', '--rigctl-port',
                      dest='rigctl_port',
                      type='string', default=[6400],
                      help='Port listening for rigctl commands (default 6400, can be comma separated list',
                      action='callback',
                      callback_args=(int,),
                      callback=get_comma_separated_args)
    group.add_option('--rigctl-addr', '--rigctl-address',
                      dest='rigctl_address',
                      type='string', default=None,
                      help='Address to listen on (default 127.0.0.1)')
    parser.add_option_group(group)

    parser.add_option('--udp-status-port',
                      dest='udp_status_port',
                      type='int', default=None,
                      help='UDP port to listen for STATUS messages (optional)')
    parser.add_option('--station-filter',
                      dest='station_filter',
                      type='string', default=None,
                      help='Nur STATUS dieser Station verwenden (optional)')
    parser.add_option('--web-port',
                      dest='web_port',
                      type='int', default=None,
                      help='Port for status web interface (optional)')
    parser.add_option('--config',
                      dest='config',
                      type='string',
                      default=None,
                      help='Pfad zu einer JSON-Konfigurationsdatei mit Optionen')

    # Nach parser = MyParser(...) und vor (options, unused_args) = parser.parse_args()
    import os

    # Erstmal nur Defaults setzen
    json_defaults = {}
    if '--config' in sys.argv:
        idx = sys.argv.index('--config')
        if len(sys.argv) > idx+1:
            config_path = sys.argv[idx+1]
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    json_defaults = json.load(f)

    # Setze Defaults aus JSON
    for key, value in json_defaults.items():
        parser.set_defaults(**{key: value})

    (options, unused_args) = parser.parse_args()

    ## clean up OptionParser which has cyclic references
    parser.destroy()
    
    if options.krec_version:
        print('kiwiclientd v1.0')
        sys.exit()

    if options.list_sound_devices:
        print(sc.all_speakers())
        sys.exit()

    FORMAT = '%(asctime)-15s pid %(process)5d %(message)s'
    logging.basicConfig(level=logging.getLevelName(options.log_level.upper()), format=FORMAT)

    run_event = threading.Event()
    run_event.set()

    httpd = None
    if options.web_port:
        httpd = start_web_server(options.web_port)
        logging.info(f"Status web interface running on port {options.web_port}")

    options.sdt = 0
    options.dir = None
    options.sound = True
    options.no_api = False
    options.nolocal = False
    options.tstamp = False
    options.station = None
    options.filename = None
    options.test_mode = False
    options.is_kiwi_wav = False
    options.is_kiwi_tdoa = False
    options.wf_cal = None
    options.netcat = False
    options.wideband = False

    gopt = options
    multiple_connections,options = options_cross_product(options)

    snd_recorders = []
    for i,opt in enumerate(options):
        opt.multiple_connections = multiple_connections
        opt.idx = i
        recorder = KiwiSoundRecorder(opt)
        worker_cls = WebKiwiWorker if opt.web_port else KiwiWorker
        snd_recorders.append(worker_cls(args=(recorder,opt,False,run_event)))
        # Starte UDP-Listener-Thread, falls gewünscht
        if getattr(opt, "udp_status_port", None):
            t = threading.Thread(
                target=udp_status_listener,
                args=(recorder, opt.udp_status_port, getattr(opt, "station_filter", None)),
                daemon=True
            )
            t.start()

    try:
        for i,r in enumerate(snd_recorders):
            if opt.launch_delay != 0 and i != 0 and options[i-1].server_host == options[i].server_host:
                time.sleep(opt.launch_delay)
            r.start()
            #logging.info("started kiwi client %d, timestamp=%d" % (i, options[i].ws_timestamp))
            logging.info("started kiwi client %d" % i)

        while run_event.is_set():
            time.sleep(.1)

    except KeyboardInterrupt:
        run_event.clear()
        join_threads(snd_recorders)
        if httpd:
            httpd.shutdown()
        print("KeyboardInterrupt: threads successfully closed")
    except Exception as e:
        print_exc()
        run_event.clear()
        join_threads(snd_recorders)
        if httpd:
            httpd.shutdown()
        print("Exception: threads successfully closed")

if __name__ == '__main__':
    #import faulthandler
    #faulthandler.enable()
    main()
# EOF
