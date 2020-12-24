import argparse
import pickle
import numpy as np

from skimage.io import imsave

from gabor_transform import GaborWaveletTransform

A0_FREQ_HZ = 27.5
HALF_STEPS_ABOVE_A = {"A": 0, "B": 2, "C": 3, "D": 5, "E": 7, "F": 8, "G": 10}

# TODO: provide a way to input these at runtime
NOTE_COLORS = np.array(
    [
        [0, 255, 0],  # A: green
        [0, 0, 255],  # Bb: blue
        [0, 255, 0],  # B: green
        [255, 0, 0],  # C: red
        [0, 0, 255],  # Db: blue
        [255, 0, 0],  # D: red
        [0, 0, 255],  # Eb: blue
        [255, 0, 0],  # E: red
        [0, 255, 0],  # F: green
        [0, 0, 255],  # Gb: blue
        [0, 255, 0],  # G: green
        [0, 0, 255],  # Ab: blue
    ]
)


def closest_geometrically(arr, target):
    log_diff = np.log(arr) - np.log(target)
    arg_closest = np.argmin(np.abs(log_diff))
    abs_log_error = np.abs(log_diff[arg_closest])
    return arg_closest, abs_log_error


def closest(arr, target):
    diff = arr - target
    arg_closest = np.argmin(np.abs(diff))
    abs_error = np.abs(diff[arg_closest])
    return arg_closest, abs_error


class TimeGrid:
    def __init__(self, start_time, bpm, beats_per_measure=None):
        self.start_time = start_time
        self.bpm = bpm
        self.beats_per_measure = beats_per_measure

    @classmethod
    def from_cli_arg(cls, time_grid_str):
        components = time_grid_str.split(",")
        if len(components) > 3 or len(components) < 2:
            raise ValueError(
                "Time grid string must have 2 or 3 comma-separated components: {time_grid_str!r} is invalid."
            )
        if len(components) == 3:
            beats_per_measure = int(components[2])
        else:
            beats_per_measure = None
        start_time = parse_time_seconds(components[0])
        bpm = float(components[1])
        return cls(start_time=start_time, bpm=bpm, beats_per_measure=beats_per_measure)


def parse_time_seconds(time_str):
    hms_components = np.array([float(component) for component in time_str.split(":")])
    return sum(
        60 ** (len(hms_components) - 1 - i) * component
        for i, component in enumerate(hms_components)
    )


def parse_pitch(pitch_str):
    try:
        return float(pitch_str)
    except ValueError:
        if len(pitch_str) < 2 or len(pitch_str) > 3:
            raise ValueError(f"Invalid pitch: {pitch_str!r}")
        note = pitch_str[0].upper()
        half_steps_above_a = HALF_STEPS_ABOVE_A.get(note)
        if half_steps_above_a is None:
            raise ValueError(f"Invalid note in pitch: {pitch_str[0]!r}")
        if len(pitch_str) == 3:
            if pitch_str[1] == "#":
                half_steps_above_a += 1
            elif pitch_str[1] == "b":
                half_steps_above_a -= 1
            else:
                raise ValueError(f"Invalid accidental in pitch: {pitch_str[1]!r}")
        try:
            octave = int(pitch_str[-1])
        except ValueError:
            raise ValueError(f"Invalid octave in pitch: {pitch_str[-1]!r}")

    return A0_FREQ_HZ * 2 ** (octave + half_steps_above_a / 12)


def shift_with_padding(arr, shift):
    if shift < 0:
        return np.pad(arr, pad_width=[[0, -shift], [0, 0]])[-shift:, :]
    elif shift > 0:
        return np.pad(arr, pad_width=[[shift, 0], [0, 0]])[:-shift, :]
    else:
        return arr


# TODO: refactor into smaller functions
def render_transform(
    transform,
    height,
    width,
    horizontal=False,
    color_pitches=True,
    start_time=None,
    end_time=None,
    min_freq=None,
    max_freq=None,
    pitch_tick_origin=None,
    time_grid=None,
    gain=1,
):
    if horizontal:
        height, width = width, height

    sample_delta = np.array(transform.strides).min()
    # all time deltas must be integer multiples of the smallest
    for delta in transform.strides:
        assert delta % sample_delta == 0

    max_len = 0
    expanded_partial_transforms = []
    full_transform_freqs = []
    for delta, partial_transform, octave_freqs in zip(
        transform.strides, transform.partial_transforms, transform.freqs
    ):
        valid_freq_mask = np.ones((len(octave_freqs)), dtype=np.bool)  # all initially True
        if min_freq is not None:
            valid_freq_mask &= octave_freqs >= min_freq
        if max_freq is not None:
            valid_freq_mask &= octave_freqs <= max_freq
        if not np.any(valid_freq_mask):
            continue

        # put all partial transforms on same time scale
        expansion_factor = delta / sample_delta
        expanded = np.repeat(partial_transform, expansion_factor, axis=0)

        expanded = expanded[:, valid_freq_mask]

        # center first sample at t=0
        shift_amt = int(expansion_factor // 2)
        expanded_partial_transforms.append(shift_with_padding(expanded, -shift_amt))

        max_len = max(max_len, expanded.shape[0])

        full_transform_freqs.extend(octave_freqs[valid_freq_mask])

    full_transform = np.hstack(
        [
            np.pad(expanded, [[0, max_len - expanded.shape[0]], [0, 0]])
            for expanded in expanded_partial_transforms
        ]
    )
    if end_time is not None:
        end_stride = int(end_time * transform.sampling_rate_hz / sample_delta)
        full_transform = full_transform[:end_stride, :]
    if start_time is not None:
        start_stride = int(start_time * transform.sampling_rate_hz / sample_delta)
        full_transform = full_transform[start_stride:, :]

    current_height = full_transform.shape[0]  # time dimension
    if current_height > height:
        pad_amt = height * int(np.ceil(current_height / height)) - current_height
        full_transform = np.pad(full_transform, [[0, pad_amt], [0, 0]])
        full_transform = full_transform.reshape(height, -1, full_transform.shape[1]).mean(axis=1)
        new_sample_delta = np.ceil(current_height / height) * sample_delta
    else:
        height_expansion_factor = int(round(height / current_height))
        full_transform = np.repeat(full_transform, height_expansion_factor, axis=0)
        new_sample_delta = sample_delta / height_expansion_factor

    image = full_transform / full_transform.max()

    if color_pitches:
        half_steps_above_a = (np.log2(np.array(full_transform_freqs) / A0_FREQ_HZ) % 1) * 12
        freq_closest_notes = np.round(half_steps_above_a).astype(int) % 12
        freq_colors = NOTE_COLORS[freq_closest_notes, :]
        image = np.expand_dims(image, axis=2) * np.expand_dims(freq_colors, axis=0)
    else:
        image = image * 255

    width_expansion_factor = max(int(round(width / image.shape[1])), 1)
    image = np.repeat(image, width_expansion_factor, axis=1)

    # These grid drawing blocks of code could be refactored to O(n) from the current O(n^2) if
    # performance becomes an issue
    if pitch_tick_origin is not None:
        max_pitch = np.max(full_transform_freqs)
        freq_log_epsilon = np.abs(np.log(full_transform_freqs[1]) - np.log(full_transform_freqs[0]))
        pitch_tick = pitch_tick_origin
        closest_freq_idx, abs_log_error = closest_geometrically(full_transform_freqs, pitch_tick)
        # find first pitch tick in displayed range
        while abs_log_error >= freq_log_epsilon:
            pitch_tick *= 2
            closest_freq_idx, abs_log_error = closest_geometrically(
                full_transform_freqs, pitch_tick
            )
            if pitch_tick > max_pitch:
                break
        # iterate over pitch ticks in displayed range
        while abs_log_error < freq_log_epsilon:
            # draw line
            line_col = int((closest_freq_idx + 0.5) * width_expansion_factor)
            image[:, line_col, ...] = np.maximum(image[:, line_col, ...], 48)  # dark gray line

            pitch_tick *= 2
            closest_freq_idx, abs_log_error = closest_geometrically(
                full_transform_freqs, pitch_tick
            )

    if time_grid is not None:
        time = new_sample_delta * np.arange(image.shape[0]) / transform.sampling_rate_hz
        if start_time is not None:
            time += start_time
        max_time = time.max()

        time_epsilon = new_sample_delta / transform.sampling_rate_hz
        time_tick = time_grid.start_time
        beat_idx = 0
        closest_time_idx, abs_error = closest(time, time_tick)
        # find first time tick in displayed range
        while abs_error >= time_epsilon:
            time_tick += 60 / time_grid.bpm  # 60 (s/min) / ... (beats/min) -> s/beat
            beat_idx += 1
            closest_time_idx, abs_error = closest(time, time_tick)
            if time_tick > max_time:
                break
        # iterate over time ticks in displayed range
        while abs_error < time_epsilon:
            if time_grid.beats_per_measure is None or beat_idx % time_grid.beats_per_measure == 0:
                intensity = 48
            else:
                intensity = 16
            image[closest_time_idx, ...] = np.maximum(image[closest_time_idx, ...], intensity)

            time_tick += 60 / time_grid.bpm  # 60 (s/min) / ... (beats/min) -> s/beat
            beat_idx += 1
            closest_time_idx, abs_error = closest(time, time_tick)

    image = np.clip(image * gain, 0, 255)
    image = image.astype(np.uint8)

    if horizontal:
        axis_order = (1, 0, 2) if color_pitches else (1, 0)
        image = image.transpose(axis_order)[::-1, ...]

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Render Gabor wavelet transform as image.

All timestamps specified through this CLI can be specified in hh:mm:ss.xyz/mm:ss.xyz format
(e.g. 1:23.456), or simply a floating point number of seconds.

All pitch values can be specified with their note names in the format [A-G](#|b)?[0-9] (e.g. F#7),
or as a floating point frequency in Hz.
    """
    )
    parser.add_argument("input_filename", help="transform file to process")
    parser.add_argument("output_filename", help="image filename to write rendered transform to")
    parser.add_argument(
        "--horizontal",
        action="store_true",
        help="show the time axis horizontally instead of vertically",
    )
    parser.add_argument(
        "--color-pitches",
        action="store_true",
        help="color the transform based on pitch",
    )
    parser.add_argument("--height", type=int, default=800, help="desired height for rendered image")
    parser.add_argument("--width", type=int, default=500, help="desired width for rendered image")
    parser.add_argument(
        "--start-time",
        help="timestamp in audio to start rendering at",
    )
    parser.add_argument(
        "--end-time",
        help="timestamp in audio to start rendering at",
    )
    parser.add_argument(
        "--min-freq",
        help="minimum pitch value to show in transform",
    )
    parser.add_argument(
        "--max-freq",
        help="maximum pitch value to show in transform",
    )
    parser.add_argument(
        "--pitch-tick-origin", help="pitch value to start per-octave pitch ticks at"
    )
    parser.add_argument(
        "--time-grid",
        help=(
            "specification to display time grid in the format "
            + '"<starting timestamp>,<bpm>[,<beats per measure>]"'
        ),
    )
    # TODO: implement more sophisticated contrast boosting e.g. using log scale or gamma
    parser.add_argument(
        "--gain",
        type=float,
        default=1,
        help=(
            "factor to multiply image intensity by before writing to file"
            + " (intensities > max intensity / gain will be clipped at maximum brightness)"
        ),
    )
    args = parser.parse_args()

    with open(args.input_filename, "rb") as f:
        transform = pickle.load(f)

    start_time = (
        None if not isinstance(args.start_time, str) else parse_time_seconds(args.start_time)
    )
    end_time = None if not isinstance(args.end_time, str) else parse_time_seconds(args.end_time)
    min_freq = None if not isinstance(args.min_freq, str) else parse_pitch(args.min_freq)
    max_freq = None if not isinstance(args.max_freq, str) else parse_pitch(args.max_freq)
    pitch_tick_origin = (
        None if not isinstance(args.pitch_tick_origin, str) else parse_pitch(args.pitch_tick_origin)
    )
    time_grid = (
        None if not isinstance(args.time_grid, str) else TimeGrid.from_cli_arg(args.time_grid)
    )

    img = render_transform(
        transform,
        horizontal=args.horizontal,
        height=args.height,
        width=args.width,
        color_pitches=args.color_pitches,
        start_time=start_time,
        end_time=end_time,
        min_freq=min_freq,
        max_freq=max_freq,
        pitch_tick_origin=pitch_tick_origin,
        time_grid=time_grid,
        gain=args.gain,
    )
    imsave(args.output_filename, img)
