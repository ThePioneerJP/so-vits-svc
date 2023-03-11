import io
import logging
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile

from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")



def main():
    import argparse

    parser = argparse.ArgumentParser(description='sovits4 inference')

    # necessary args
    parser.add_argument('-m', '--model_path', type=str, default="logs/44k/G_0.pth", help='model path')
    parser.add_argument('-c', '--config_path', type=str, default="configs/config.json", help='config.json path')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["君の知らない物語-src.wav"], help='wav file name. put your audio file under the raw folder.')
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0], help='pitch. supports both plus and minus (half step per pitch).')
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['nen'], help='The name of the person you want to convert to.')

    # optional args
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False,
                        help='auto prediction. do not turn it on when converting a song, as it will go out of tune.')
    parser.add_argument('-cm', '--cluster_model_path', type=str, default="logs/44k/kmeans_10000.pt", help='The path of the clustering model, if there is no training cluster, just ignore it.')
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=0, help='Proportion of clustering scheme, range 0-1, if no clustering model is trained, ignore it, as the default value 0 will be used.')

    # unnecessary args
    parser.add_argument('-sd', '--slice_db', type=int, default=-40, help='Default -40, noisy audio can be -30, dry sound can hold breath -50')
    parser.add_argument('-d', '--device', type=str, default=None, help='Inference device, if None it will automatically select cpu or gpu')
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4, help='The noise level will affect the articulation and sound quality, which is more metaphysical')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5, help='Inferring the number of seconds of the audio pad, there will be abnormal noise at the beginning and end due to unknown reasons, and the pad will not appear after a short period of silence')
    parser.add_argument('-wf', '--wav_format', type=str, default='wav', help='output fotmat.')

    args = parser.parse_args()

    svc_model = Svc(args.model_path, args.config_path, args.device, args.cluster_model_path)
    infer_tool.mkdir(["raw", "results"])
    clean_names = args.clean_names
    trans = args.trans
    spk_list = args.spk_list
    slice_db = args.slice_db
    wav_format = args.wav_format
    auto_predict_f0 = args.auto_predict_f0
    cluster_infer_ratio = args.cluster_infer_ratio
    noice_scale = args.noice_scale
    pad_seconds = args.pad_seconds

    infer_tool.fill_a_to_b(trans, clean_names)
    for clean_name, tran in zip(clean_names, trans):
        raw_audio_path = f"raw/{clean_name}"
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        wav_path = Path(raw_audio_path).with_suffix('.wav')
        chunks = slicer.cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

        for spk in spk_list:
            audio = []
            for (slice_tag, data) in audio_data:
                print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')

                length = int(np.ceil(len(data) / audio_sr * svc_model.target_sample))
                if slice_tag:
                    print('jump empty segment')
                    _audio = np.zeros(length)
                else:
                    # padd
                    pad_len = int(audio_sr * pad_seconds)
                    data = np.concatenate([np.zeros([pad_len]), data, np.zeros([pad_len])])
                    raw_path = io.BytesIO()
                    soundfile.write(raw_path, data, audio_sr, format="wav")
                    raw_path.seek(0)
                    out_audio, out_sr = svc_model.infer(spk, tran, raw_path,
                                                        cluster_infer_ratio=cluster_infer_ratio,
                                                        auto_predict_f0=auto_predict_f0,
                                                        noice_scale=noice_scale
                                                        )
                    _audio = out_audio.cpu().numpy()
                    pad_len = int(svc_model.target_sample * pad_seconds)
                    _audio = _audio[pad_len:-pad_len]

                audio.extend(list(infer_tool.pad_array(_audio, length)))
            key = "auto" if auto_predict_f0 else f"{tran}key"
            cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            res_path = f'./results/{clean_name}_{key}_{spk}{cluster_name}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)

if __name__ == '__main__':
    main()
