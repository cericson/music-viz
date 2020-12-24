import argparse
import os
import pickle
import subprocess as sp
import time

import numpy as np
from tqdm.auto import tqdm

BASE_FREQ_HZ = 27.5  # A0
OCTAVE_RANGE = 8  # 8 * 12 = 96 unique pitches (8 higher than the range of a piano)


class GaborWaveletTransform:
    def __init__(self, partial_transforms, freqs, strides, sampling_rate_hz=44_100.0):
        self.sampling_rate_hz = sampling_rate_hz
        self.partial_transforms = partial_transforms
        self.strides = strides
        self.freqs = freqs

    @classmethod
    def from_audio(cls, audio, sampling_rate_hz=44_100.0, pitch_oversampling=1):
        audio = np.mean(audio, axis=1).astype(audio.dtype)
        assert audio.flags["C_CONTIGUOUS"]  # necessary for strided convolution
        dsize = audio.itemsize
        st = time.time()

        pitches_per_octave = 12 * pitch_oversampling

        beta = 16
        # 6 SDs from gaussian center
        base_width = 6 * beta * (sampling_rate_hz / BASE_FREQ_HZ) // 2 * 2 + 1

        octave_transforms = []
        octave_freqs = []
        octave_strides = []
        for octave in tqdm(range(OCTAVE_RANGE), desc="Computing octaves of transform"):
            filter_width = int((base_width / 2 ** octave) // 2 * 2 + 1)
            octave_base_freq = BASE_FREQ_HZ * 2 ** octave
            t = (
                np.arange(-(filter_width // 2), filter_width // 2 + 1)[:, np.newaxis]
                / sampling_rate_hz
            )
            freqs = octave_base_freq * 2 ** (np.arange(pitches_per_octave) / pitches_per_octave)
            octave_freqs.append(freqs)
            # doing this properly with complex numbers takes about 2x as long in numpy for some reason
            # TODO: try removing DC component of wavelets (subtract gaussian with same width to make
            # sum of filter 0)
            # TODO: turn filter bank generation into func
            # TODO (perf): compare speed computing one conv at a time
            wbank = np.hstack(
                [
                    freqs
                    / sampling_rate_hz
                    * fn(2 * np.pi * t * freqs)
                    * np.exp(-((t * freqs / beta) ** 2))
                    for fn in [np.sin, np.cos]
                ]
            ).astype(np.float32)

            oct_scale = 2 ** (OCTAVE_RANGE - octave - 1)
            skip_len = round(sampling_rate_hz / octave_base_freq * beta / 2 / oct_scale) * oct_scale
            octave_strides.append(skip_len)
            # ensure initial conv value has wavelet centered at t=0
            padded = np.pad(audio, filter_width // 2)
            n_skip = (len(padded) - filter_width) // skip_len + 1
            # TODO (perf): try downsampling before convolution for more speed?
            # TODO: turn strided convolution with multiple kernels at once into func
            view = np.lib.stride_tricks.as_strided(
                padded, shape=(n_skip, filter_width), strides=(dsize * skip_len, dsize)
            )
            res = view.dot(wbank)
            res = np.sqrt(res[:, :pitches_per_octave] ** 2 + res[:, -pitches_per_octave:] ** 2)
            octave_transforms.append(res)

        print(f"Elapsed: {time.time() - st:.3f} s")
        return cls(
            octave_transforms, octave_freqs, octave_strides, sampling_rate_hz=sampling_rate_hz
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Gabor wavelet transform.")
    parser.add_argument("input_filename", help="audio file to process")
    parser.add_argument("output_filename", help="filename to write transform to")
    parser.add_argument("--channels", type=int, default=2, help="number of channels in audio data")
    parser.add_argument(
        "--sampling_rate_hz", type=float, default=44_100.0, help="sampling rate of audio"
    )
    parser.add_argument("--bit_depth", type=int, default=16, help="audio bit depth")
    parser.add_argument(
        "--pitch-oversampling", type=int, default=1, help="pitches to sample per half step"
    )
    args = parser.parse_args()

    assert args.bit_depth in {8, 16, 32}

    command = [
        "ffmpeg",
        "-i",
        args.input_filename,
        "-f",
        f"s{args.bit_depth}le",
        "-acodec",
        f"pcm_s{args.bit_depth}le",
        "-ar",
        str(args.sampling_rate_hz),
        "-ac",
        str(args.channels),
        "-",
    ]

    print("Loading audio data...")
    with open("ffmpeg_log.txt", "w") as fp:
        pipe = sp.Popen(command, stdout=sp.PIPE, stderr=fp, bufsize=10 ** 8)

    raw_bytes = pipe.stdout.read()
    audio = np.fromstring(raw_bytes, dtype=f"int{args.bit_depth}")
    audio = audio.reshape((audio.shape[0] // args.channels, args.channels))

    transform = GaborWaveletTransform.from_audio(
        audio, sampling_rate_hz=args.sampling_rate_hz, pitch_oversampling=args.pitch_oversampling
    )
    with open(args.output_filename, "wb") as f:
        pickle.dump(transform, f)
