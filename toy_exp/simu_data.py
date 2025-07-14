#For mix2both -> librimix dataset (speakers, wham_noise)
#Simulate vad, speech, speaker, noise, mix

import json
import random
from pathlib import Path
from glob import glob
import os
import yaml
import soundfile as sf

# 1. Define your tool schemas
TOOLS = [
    {
        "name": "asr_transcribe",
        "description": "Convert speech in the audio file to text",
        "parameters": {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"}
            },
            "required": ["audio_path"]
        }
    },
    {
        "name": "translator",
        "description": "Translate text from one language to another",
        "parameters": {
            "type": "object",
            "properties": {
                "text":       {"type": "string"},
                "source_lang":{"type": "string"},
                "target_lang":{"type": "string"}
            },
            "required": ["text","target_lang"]
        }
    },
    {
        "name": "tts",
        "description": "Convert text to speech",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "speaker_diarization",
        "description": "Diarize the speakers in the audio",
        "parameters": {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"}
            },
            "required": ["audio_path"]
        }
    },
    {
        "name": "source_separation",
        "description": "Separate the speakers in the audio",
        "parameters": {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"}
            },
            "required": ["audio_path"]
        }
    },
    {
        "name": "estimate_audio_quality",
        "description": "Estimate the quality of the audio, and if it is good enough for transcription",
        "parameters": {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"}
            },
            "required": ["audio_path"]
        }
    },
    {
        "name": "remove_noise_distortion",
        "description": "Remove noise and distortion from the audio",
        "parameters": {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"}
            },
            "required": ["audio_path"]
        }
    },
    {
        "name": "speaker_identification",
        "description": "Identify the speakers in the audio",
        "parameters": {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"}
            },
            "required": ["audio_path"]
        }
    }
]


Librispeech_root="/ocean/projects/cis210027p/shared/corpora/librispeech/LibriSpeech"
mix2_audio_wavs = glob('/ocean/projects/cis210027p/shared/corpora/librimix/Libri2Mix/wav_16k/max/train_100/mix_both/*.wav')
mix2_audio_wavs = random.sample(mix2_audio_wavs, 100)

# 2. Define a few toy audio paths & transcripts
AUDIO_CLIPS_MIX2 = []

for mix2_wav in mix2_audio_wavs:
    utt_1, utt_2 = os.path.basename(mix2_wav).replace('.wav', '').split('_')
    folder_path1 = utt_1.split('-')
    folder_path2 = utt_2.split('-')
    utt_1_tans_dict = {}
    utt_2_tans_dict = {}
    with open('{}/train-clean-100/{}/{}/{}-{}.trans.txt'.format(Librispeech_root, folder_path1[0], folder_path1[1], folder_path1[0], folder_path1[1])) as f:
        for line in f:
            utt_id, text = line.strip().split(' ', 1)
            utt_1_tans_dict[utt_id] = text.lower()
    with open('{}/train-clean-100/{}/{}/{}-{}.trans.txt'.format(Librispeech_root, folder_path2[0], folder_path2[1], folder_path2[0], folder_path2[1])) as f:
        for line in f:
            utt_id, text = line.strip().split(' ', 1)
            utt_2_tans_dict[utt_id] = text.lower()
    AUDIO_CLIPS_MIX2.append({
        "path": mix2_wav,
        "src1_path": '{}/train-clean-100/{}/{}/{}.flac'.format(Librispeech_root, folder_path1[0], folder_path1[1], utt_1),
        "src2_path": '{}/train-clean-100/{}/{}/{}.flac'.format(Librispeech_root, folder_path2[0], folder_path2[1], utt_2),
        "lang": "en",
        "transcript_1": utt_1_tans_dict[utt_1],
        "transcript_2": utt_2_tans_dict[utt_2],
        "noise": "Noisy"
    })


multi_tools_variants = ["Tell me what are these people saying?", "Can you make out what everyone is saying in this noisy recording?", \
    "Despite the background noise, please transcribe what each person is saying.", "Is it possible to separate the voices and tell me what was said?", \
    "This audio is pretty muddled, but I need to know what the different speakers are saying.", "Given the chatter and ambient sound, can you extract and list what each individual contributed?"]
# 3. Milestone logic: for each clip decide which tools to call
def build_conversation(meta):
    conv = []
    # 1) Always start with ASR
    conv.append({"from":"human", "value":f"<audio>{meta['path']}</audio> {random.choice(multi_tools_variants)}"})
    if meta["noise"] == "Noisy":
        conv.append({"from":"function_call", "value":json.dumps({
            "name":"estimate_audio_quality",
            "arguments":{"audio_path":meta["path"]}
        })})
        conv.append({"from":"observation", "value":json.dumps({
            "audio_quality": meta["noise"]
        })})
        conv.append({"from":"function_call", "value":json.dumps({
            "name":"remove_noise_distortion",
            "arguments":{"audio_path":meta["path"]}
        })})
        conv.append({"from":"observation", "value":json.dumps({
            "path_clean": meta["path"].replace(".wav", "_clean.wav")
        })})
        meta["path_clean"] = meta["path"].replace(".wav", "_clean.wav")
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"source_separation",
        "arguments":{"audio_path":meta.get("path_clean", meta["path"])}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "path_clean": meta["path"].replace(".wav", "_clean.wav")
    })})
    last_audio = (meta["src1_path"], meta["src2_path"])
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"asr_transcribe",
        "arguments":{"audio_path":last_audio[0]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "transcript_1": meta["transcript_1"]
    })})
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"asr_transcribe",
        "arguments":{"audio_path":last_audio[1]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "transcript_2": meta["transcript_2"]
    })})
    last_text = (meta["transcript_1"], meta["transcript_2"])

    # 5) Finally, synthesize back to audio
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"tts",
        "arguments":{"text": last_text[0]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "tts_audio_path": "audio/final_tts1.wav"
    })})
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"tts",
        "arguments":{"text": last_text[1]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "tts_audio_path": "audio/final_tts2.wav"
    })})

    # 6) Model’s wrap-up summary
    summary = (
        f"Speaker 1: “{meta['transcript_1']}”. "
        f"Speaker 2: “{meta['transcript_2']}”. "
        f"Final audio at ['audio/final_tts1.wav', 'audio/final_tts2.wav']."
    )
    conv.append({"from":"gpt", "value": summary})
    return conv

# 4. Build the toy dataset
random.seed(42)
dataset = []
for clip in AUDIO_CLIPS_MIX2:
    sample = {
        "conversations": build_conversation(clip),
        "tools": json.dumps(TOOLS)
    }
    dataset.append(sample)

asr_vairants = ["Convert this to text.", "Transcribe the audio.", "Please generate a transcript of this recording.", "Can you write down what's being said here?", \
    "Start transcription.", "I need a written version of this.", "Get me the text from this audio.", "Process this for transcription.", "Dictate this audio.", "Create a text file from this sound."]

#Single tool scenarios, ASR, ST, SER, SE
def build_asr_no_se_conversation(meta):
    conv = []
    # 1) Always start with ASR
    conv.append({"from":"human", "value":f"<audio>{meta['path']}</audio> {random.choice(asr_vairants)}"})
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"estimate_audio_quality",
        "arguments":{"audio_path":meta["path"]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "audio_quality": 'Clean'
    })})
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"asr_transcribe",
        "arguments":{"audio_path":meta["path"]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "transcript": meta["transcript"]
    })})
    summary = (
        f"Transcript: “{meta['transcript']}”. "
    )
    conv.append({"from":"gpt", "value": summary})
    return conv

asr_wavs = glob(f'{Librispeech_root}/train-clean-360/**/*.flac')
asr_wavs = random.sample(asr_wavs, 100)
AUDIO_CLIPS_ASR_no_SE = []
utt_dict = {}
for wav in asr_wavs:
    utt_id = os.path.basename(wav).replace('.wav', '')
    folder_path = utt_id.split('-')
    with open('{}/train-clean-360/{}/{}/{}-{}.trans.txt'.format(Librispeech_root, folder_path[0], folder_path[1], folder_path[0], folder_path[1])) as f:
        for line in f:
            utt_id, text = line.strip().split(' ', 1)
            utt_dict[utt_id] = text.lower()
    AUDIO_CLIPS_ASR_no_SE.append({
        "path": wav,
        "transcript": utt_dict[utt_id]
    })

for clip in AUDIO_CLIPS_ASR_no_SE:
    sample = {
        "conversations": build_asr_no_se_conversation(clip),
        "tools": json.dumps(TOOLS)
    }
    dataset.append(sample)

#chime4/CHiME3/data/audio/16kHz/isolated_1ch_track/*05_*_real/*.wav
#chime4/CHiME3/data/transcriptions/dt05_bus_real/F01_22GC010X_BUS.trn -> utt_id text
def build_asr_se_conversation(meta):
    conv = []
    # 1) Always start with ASR
    conv.append({"from":"human", "value":f"<audio>{meta['path']}</audio> {random.choice(asr_vairants)}"})
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"estimate_audio_quality",
        "arguments":{"audio_path":meta["path"]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "audio_quality": 'Noisy'
    })})
    conv.append({"from":"function_call", "value":json.dumps({
            "name":"remove_noise_distortion",
            "arguments":{"audio_path":meta["path"]}
        })})
    conv.append({"from":"observation", "value":json.dumps({
        "path_clean": meta["path"].replace(".wav", "_clean.wav")
    })})
    meta["path_clean"] = meta["path"].replace(".wav", "_clean.wav")
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"asr_transcribe",
        "arguments":{"audio_path":meta["path_clean"]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "transcript": meta["transcript"]
    })})
    summary = (
        f"Transcript: “{meta['transcript']}”. "
    )
    conv.append({"from":"gpt", "value": summary})
    return conv

chime_root = '/ocean/projects/cis210027p/shared/corpora/chime4/CHiME3'
chime_wavs = glob(f'{chime_root}/data/audio/16kHz/isolated_1ch_track/*05_*_real/*.wav')
chime_wavs = random.sample(chime_wavs, 100)
AUDIO_CLIPS_ASR = []
utt_dict = {}
for wav in chime_wavs:
    utt_id = os.path.basename(wav).replace('.wav', '')
    folder_name = wav.split('/')[-2]
    with open('{}/transcriptions/{}/{}.trn'.format(chime_root, folder_name, utt_id)) as f:
        for line in f:
            utt_id, text = line.strip().split(' ', 1)
            utt_dict[utt_id] = text.lower()
    AUDIO_CLIPS_ASR.append({
        "path": wav,
        "transcript": utt_dict[utt_id]
    })

for clip in AUDIO_CLIPS_ASR:
    sample = {
        "conversations": build_asr_se_conversation(clip),
        "tools": json.dumps(TOOLS)
    }
    dataset.append(sample)


lang_map = {'ar': 'Arabic', 'de': 'German', 'en': 'English', 'es': 'Spanish', 'fr': 'French', 'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'fa': 'Persian', 'nl': 'Dutch', 'ro': 'Romanian'}
st_en2x_vairants = ["Translate this audio message to ", "Convert this spoken message into ", "Provide a translation of this audio into ", \
    "Translate what's being said here into ", "Give me this audio in "]
def build_st_en2x_conversation(meta):
    conv = []
    # 1) Always start with ASR
    conv.append({"from":"human", "value":f"<audio>{meta['path']}</audio> {random.choice(st_en2x_vairants)}{lang_map[meta['lang']]}"})
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"estimate_audio_quality",
        "arguments":{"audio_path":meta["path"]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "audio_quality": 'Clean'
    })})
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"asr_transcribe",
        "arguments":{"audio_path":meta["path"]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "transcript": meta["transcript"]
    })})
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"translator",
        "arguments":{"text":meta["transcript"], "source_lang":"en", "target_lang":meta["lang"]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "translated_text": meta["translated_text"]
    })})
    summary = (
        f"Translated text: “{meta['translated_text']}”. "
    )
    conv.append({"from":"gpt", "value": summary})
    return conv

must_c_root = '/ocean/projects/cis210027p/shared/corpora/must-c_v1.2'
must_c_wavs = glob(f'{must_c_root}/en-*/data/train/wav/*.wav')
must_c_wavs = random.sample(must_c_wavs, 100)
AUDIO_CLIPS_ST = []
for wav_file in must_c_wavs:
    # extract lang‐pair and target code
    lang_pair = os.path.basename(os.path.dirname(os.path.dirname(wav_file)))  # e.g. "en-ar"
    tgt_lang  = lang_pair.split('-')[1]

    txt_dir = os.path.join(must_c_root, lang_pair, 'data', 'train', 'txt')
    
    # 1) load all segments for this lang‐pair
    with open(os.path.join(txt_dir, 'train.yaml')) as yf:
        all_segments = yaml.safe_load(yf)

    # 2) load transcripts as simple line lists
    with open(os.path.join(txt_dir, 'train.en')) as ef:
        en_lines = [l.strip().lower() for l in ef]
    with open(os.path.join(txt_dir, f'train.{tgt_lang}')) as tf:
        tr_lines = [l.strip()     for l in tf]

    # 3) pick only those segments from this specific wav
    basename = os.path.basename(wav_file)  # e.g. "ted_1.wav"
    indexed_segs = [
        (idx, seg) 
        for idx, seg in enumerate(all_segments) 
        if seg['wav'] == basename
    ]
    if not indexed_segs:
        continue  # no segments found for this file

    # 4) choose one at random
    idx, seg = random.choice(indexed_segs)

    # attach transcripts by that global index
    seg['transcript']       = en_lines[idx]
    seg['translated_text']  = tr_lines[idx]
    seg['lang']             = tgt_lang

    # 5) read & slice
    data, sr = sf.read(wav_file)
    start = int(seg['offset'] * sr)
    end   = start + int(seg['duration'] * sr)
    clip  = data[start:end]

    # 6) write out
    out_dir = 'must_c_ST_segments'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{basename}_{lang_pair}_{idx}.wav")
    sf.write(out_path, clip, sr)
    seg['path'] = out_path
    AUDIO_CLIPS_ST.append({
        "path": out_path,
        "transcript": seg['transcript'],
        "translated_text": seg['translated_text'],
        "lang": tgt_lang
    })

for seg in AUDIO_CLIPS_ST:
    sample = {
        "conversations": build_st_en2x_conversation(seg),
        "tools": json.dumps(TOOLS)
    }
    dataset.append(sample)


st_en2x_tts_vairants = ["Translate this audio message to [LANGUAGE], and then speak it back in a different voice.", "Can you translate this audio into [LANGUAGE] and play it back with a different voice?", \
    "I need this audio translated to [LANGUAGE] and spoken aloud in an alternative voice.", "Translate and re-voice this audio into [LANGUAGE].", "Translate this message to [LANGUAGE] and use a distinct voice for playback."]
def build_st_en2x_tts_conversation(meta):
    conv = []
    # 1) Always start with ASR
    conv.append({"from":"human", "value":f"<audio>{meta['path']}</audio> {random.choice(st_en2x_tts_vairants)}".replace("[LANGUAGE]", lang_map[meta['lang']])})
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"estimate_audio_quality",
        "arguments":{"audio_path":meta["path"]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "audio_quality": 'Clean'
    })})
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"asr_transcribe",
        "arguments":{"audio_path":meta["path"]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "transcript": meta["transcript"]
    })})
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"translator",
        "arguments":{"text":meta["transcript"], "source_lang":"en", "target_lang":meta["lang"]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "translated_text": meta["translated_text"]
    })})
    conv.append({"from":"function_call", "value":json.dumps({
        "name":"tts",
        "arguments":{"text":meta["translated_text"]}
    })})
    conv.append({"from":"observation", "value":json.dumps({
        "tts_audio_path": "audio/final_tts.wav"
    })})
    summary = (
         f"Final audio at 'audio/final_tts.wav'."
    )
    conv.append({"from":"gpt", "value": summary})
    return conv

must_c_root = '/ocean/projects/cis210027p/shared/corpora/must-c_v1.2'
must_c_wavs = glob(f'{must_c_root}/en-*/data/train/wav/*.wav')
must_c_wavs = random.sample(must_c_wavs, 100)
AUDIO_CLIPS_ST_TTS = []
for wav_file in must_c_wavs:
    # extract lang‐pair and target code
    lang_pair = os.path.basename(os.path.dirname(os.path.dirname(wav_file)))  # e.g. "en-ar"
    tgt_lang  = lang_pair.split('-')[1]

    txt_dir = os.path.join(must_c_root, lang_pair, 'data', 'train', 'txt')
    
    # 1) load all segments for this lang‐pair
    with open(os.path.join(txt_dir, 'train.yaml')) as yf:
        all_segments = yaml.safe_load(yf)

    # 2) load transcripts as simple line lists
    with open(os.path.join(txt_dir, 'train.en')) as ef:
        en_lines = [l.strip().lower() for l in ef]
    with open(os.path.join(txt_dir, f'train.{tgt_lang}')) as tf:
        tr_lines = [l.strip()     for l in tf]

    # 3) pick only those segments from this specific wav
    basename = os.path.basename(wav_file)  # e.g. "ted_1.wav"
    indexed_segs = [
        (idx, seg) 
        for idx, seg in enumerate(all_segments) 
        if seg['wav'] == basename
    ]
    if not indexed_segs:
        continue  # no segments found for this file

    # 4) choose one at random
    idx, seg = random.choice(indexed_segs)

    # attach transcripts by that global index
    seg['transcript']       = en_lines[idx]
    seg['translated_text']  = tr_lines[idx]
    seg['lang']             = tgt_lang

    # 5) read & slice
    data, sr = sf.read(wav_file)
    start = int(seg['offset'] * sr)
    end   = start + int(seg['duration'] * sr)
    clip  = data[start:end]

    # 6) write out
    out_dir = 'must_c_ST_TTS_segments'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{basename}_{lang_pair}_{idx}.wav")
    sf.write(out_path, clip, sr)
    seg['path'] = out_path
    AUDIO_CLIPS_ST_TTS.append({
        "path": out_path,
        "transcript": seg['transcript'],
        "translated_text": seg['translated_text'],
        "lang": tgt_lang
    })

for seg in AUDIO_CLIPS_ST_TTS:
    sample = {
        "conversations": build_st_en2x_tts_conversation(seg),
        "tools": json.dumps(TOOLS)
    }
    dataset.append(sample)
# 5. Write to JSON file
out_path = Path("toy_audio_toolcall.json")
with out_path.open("w") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Generated {len(dataset)} samples → {out_path}")
