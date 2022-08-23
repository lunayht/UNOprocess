import argparse
import os
from typing import List, Union

import librosa
import soundfile


def resample_wav(
    root: str, ori_data_list: List[str], target_data_path: str, sample_rate: int
) -> None:
    r"""Resamples wav files to the target sample rate.

    Args:
        root: Path to the root directory of the original .wav data.
        ori_data_list: List of original .wav filenames.
        target_data_path: Target path to save resampled .wav files.
        sample_rate: Target sample rate of the audio.
    """
    for ori_wav_name in ori_data_list:
        ori_wav_path = os.path.join(root, ori_wav_name)
        out_path = os.path.join(target_data_path, ori_wav_name)
        ori_wav, _ = librosa.load(ori_wav_path, sr=sample_rate, res_type="kaiser_fast")
        soundfile.write(out_path, ori_wav, samplerate=sample_rate)
        print("\t", out_path)


def slice_wav_into_breathing_cycle(
    root: str,
    ori_data_list: List[str],
    target_data_path: str,
    ann_txt_path: str,
    sample_rate: int,
) -> None:
    r"""Slice raw waveform into small chunks according to breathing cycle start-end time and extract crackle and wheeze labels.

    Args:
        root: Path to the root directory of the original .wav data.
        ori_data_list: List of original .wav filenames.
        target_data_path: Target path to save resampled .wav files.
        ann_txt_path: Path that contains original .txt annotation files.
        sample_rate: Target sample rate of the audio.

    Output:
        Save wav chunk as: `TARGETPATH/OriginalFilename_CycleCount_Crackle_Wheeze.wav`.

        For example,
        `226_1b1_Pl_sc_LittC2SE_2_10.wav` represents the second cycle of the original `226_1b1_Pl_sc_LittC2SE.wav`, with crackle = 1, wheeze = 0.
    """
    for ori_wav_name in ori_data_list:
        ori_wav_path = os.path.join(root, ori_wav_name)
        ori_wav, _ = librosa.load(ori_wav_path, sr=sample_rate, res_type="kaiser_fast")
        ori_ann = os.path.join(ann_txt_path, ori_wav_name.replace("wav", "txt"))
        ann_list = open(ori_ann, "r").readlines()

        for cycle_count, annotation in enumerate(ann_list):
            marker = annotation.replace("\n", "").split("\t")
            start = int(float(marker[0]) * sample_rate)
            end = int(float(marker[1]) * sample_rate)
            crackle = int(marker[2])
            wheeze = int(marker[3])
            wav_chunk = ori_wav[start : end + 1]
            chunk_name = f'{ori_wav_name.replace(".wav", "")}_{cycle_count}_{crackle}{wheeze}.wav'
            chunk_path = os.path.join(target_data_path, chunk_name)
            soundfile.write(chunk_path, wav_chunk, samplerate=sample_rate)


def args_parser() -> argparse.Namespace:
    r"""Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument(
        "--ori_data_path",
        type=str,
        required=True,
        metavar="PATH",
        help="Path that contains original .wav files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        metavar="PATH",
        help="Target path to save .wav chunks.",
    )
    parser.add_argument(
        "--ori_txt_path",
        type=str,
        default=None,
        metavar="PATH",
        help="path that contains original .txt annotation files. Note: can be the same folder as original .wav",
    )
    parser.add_argument(
        "--slice",
        action="store_true",
        help="slice original wav files into chunks",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=16000,
        help="Target sample rate of the audio. (default: 16000)",
    )
    parser.add_argument("--slice_sec", type=Union[float, int], default=None, help="")
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()

    ori_data_list = filter(
        lambda file: file.endswith(".wav"), os.listdir(args.ori_data_path)
    )

    if not os.path.exists(args.target_data_path):
        os.makedirs(args.target_data_path)

    if args.slice:
        if args.ori_txt_path is None:
            raise ValueError(
                "Please specify the path of original .txt annotation files."
            )
        slice_wav_into_breathing_cycle(
            root=args.ori_data_path,
            ori_data_list=ori_data_list,
            target_data_path=args.target_data_path,
            ann_txt_path=args.ori_txt_path,
            sample_rate=args.target_sample_rate,
        )
    else:
        resample_wav(
            root=args.ori_data_path,
            ori_data_list=ori_data_list,
            target_data_path=args.target_data_path,
            sample_rate=args.target_sample_rate,
        )
