from whisper_live.client import TranscriptionClient
# from whisper_live.client.Client import Client

client = TranscriptionClient(
  "localhost",
  9090,
#   lang="en",
  translate=False,
  model="small",
  use_vad=True,
  
)

# client("assets/jfk.flac")
client()

save = Client.record(out_file='output_recording1.wav')
# whisper_live.client.record(out_file='output_recording1.wav')
# server = TranscriptionClient()
# server.run("0.0.0.0",9090)