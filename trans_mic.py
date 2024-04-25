#python3 trans_mic.py --model-source speechbrain/asr-streaming-conformer-librispeech --ip=localhost --device=cpu

from argparse import ArgumentParser
from dataclasses import dataclass
import logging

parser = ArgumentParser()
parser.add_argument("--model-source", required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--ip", default="127.0.0.1")
parser.add_argument("--port", default=9431)
parser.add_argument("--chunk-size", default=24, type=int)
parser.add_argument("--left-context-chunks", default=4, type=int)
parser.add_argument("--num-threads", default=None, type=int)
parser.add_argument("--verbose", "-v", default=False, action="store_true")
args = parser.parse_args()

if args.verbose:
    logging.getLogger().setLevel(logging.INFO)

logging.info("Loading libraries")

from speechbrain.inference.ASR import StreamingASR, ASRStreamingContext
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
import torch
import gradio as gr
import torchaudio
import numpy as np

device = args.device

if args.num_threads is not None:
    torch.set_num_threads(args.num_threads)

logging.info(f"Loading model from \"{args.model_source}\" onto device {device}")

asr = StreamingASR.from_hparams(args.model_source, run_opts={"device": device})
config = DynChunkTrainConfig(args.chunk_size, args.left_context_chunks)

@dataclass
class GradioStreamingContext:
    context: ASRStreamingContext
    chunk_size: int
    waveform_buffer: torch.Tensor
    decoded_text: str

def transcribe(stream, new_chunk):
    sr, y = new_chunk

    y = y.astype(np.float32)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    y /= max(1, torch.max(torch.abs(y)).item())  # norm by max abs() within chunk & avoid NaN
    if len(y.shape) > 1:
        y = torch.mean(y, dim=1)  # downmix to mono

    # HACK: we are making poor use of the resampler across chunk boundaries
    # which may degrade accuracy.
    # NOTE: we should also absolutely avoid recreating a resampler every time
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=asr.audio_normalizer.sample_rate).to(device)
    y = resampler(y)  # janky resample (probably to 16kHz)


    if stream is None:
        stream = GradioStreamingContext(
            context=asr.make_streaming_context(config),
            chunk_size=asr.get_chunk_size_frames(config),
            waveform_buffer=y,
            decoded_text="",
        )
    else:
        stream.waveform_buffer = torch.concat((stream.waveform_buffer, y))

    while stream.waveform_buffer.size(0) > stream.chunk_size:
        chunk = stream.waveform_buffer[:stream.chunk_size]
        stream.waveform_buffer = stream.waveform_buffer[stream.chunk_size:]

        # fake batch dim
        chunk = chunk.unsqueeze(0)

        # list of transcribed strings, of size 1 because the batch size is 1
        with torch.no_grad():
            transcribed = asr.transcribe_chunk(stream.context, chunk)
        stream.decoded_text += transcribed[0]

    return stream, stream.decoded_text

# NOTE: latency seems relatively high, which may be due to this:
# https://github.com/gradio-app/gradio/issues/6526

demo = gr.Interface(
    transcribe,
    ["state", gr.Audio(sources=["microphone"], streaming=True)],
    ["state", "text"],
    live=True,
)

demo.launch(server_name=args.ip, server_port=args.port)
