from speechbrain.inference.ASR import StreamingASR
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig


asr_model = StreamingASR.from_hparams(
    source="speechbrain/asr-streaming-conformer-librispeech",
    savedir="pretrained_models/asr-streaming-conformer-librispeech"
)
text = asr_model.transcribe_file(
    "rec/rec_anger.wav",
    # select a chunk size of ~960ms with 4 chunks of left context
    DynChunkTrainConfig(24, 4),
    # disable torchaudio streaming to allow fetching from HuggingFace
    # set this to True for your own files or streams to allow for streaming file decoding
    use_torchaudio_streaming=False,
)
# for text_chunk in asr_model.transcribe_file:
    # print(text_chunk, flush=True, end="")
print("Transcribe:", text)


# python3 app1.py test-en.wav --model-source=speechbrain/asr-streaming-conformer-librispeech --device=cpu -v

# from argparse import ArgumentParser
# import logging

# parser = ArgumentParser()
# parser.add_argument("audio_path")
# parser.add_argument("--model-source", required=True)
# parser.add_argument("--device", default="cpu")
# parser.add_argument("--ip", default="127.0.0.1")
# parser.add_argument("--port", default=9431)
# parser.add_argument("--chunk-size", default=24, type=int)
# parser.add_argument("--left-context-chunks", default=4, type=int)
# parser.add_argument("--num-threads", default=None, type=int)
# parser.add_argument("--verbose", "-v", default=False, action="store_true")
# args = parser.parse_args()

# if args.verbose:
#     logging.getLogger().setLevel(logging.INFO)

# logging.info("Loading libraries")

# from speechbrain.inference.ASR import StreamingASR
# from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
# import torch

# device = args.device

# if args.num_threads is not None:
#     torch.set_num_threads(args.num_threads)

# logging.info(f"Loading model from \"{args.model_source}\" onto device {device}")

# asr = StreamingASR.from_hparams(args.model_source, run_opts={"device": device})
# config = DynChunkTrainConfig(args.chunk_size, args.left_context_chunks)

# logging.info(f"Starting stream from URI \"{args.audio_path}\"")

# for text_chunk in asr.transcribe_file_streaming(args.audio_path, config):
#     print(text_chunk, flush=True, end="")
