from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor
from whisper.audio import HOP_LENGTH, load_audio, log_mel_spectrogram, pad_or_trim


class WhisperVQDataset(Dataset):
    def __init__(
        self, filelist: str, model_name_or_path: str = "openai/whisper-medium"
    ):
        super().__init__()

        self.files = [
            Path(line.strip()) for line in Path(filelist).read_text().splitlines()
        ]
        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        wav = load_audio(file)
        wav_length = wav.shape[-1]
        mel_length = wav_length // HOP_LENGTH + 1

        wav = pad_or_trim(wav)
        wav = torch.from_numpy(wav).float()
        input_features = log_mel_spectrogram(wav)
        mel_mask = torch.zeros(input_features.shape[1], dtype=torch.float)
        mel_mask[:mel_length] = 1

        input_ids = file.with_suffix(".whisper.txt").read_text().strip().split("\t")[0]
        input_ids = [int(x) for x in input_ids.split(",")]

        while input_ids[-1] in [
            self.processor.tokenizer.pad_token_id,
            self.processor.tokenizer.eos_token_id,
        ]:
            input_ids.pop()

        input_ids.append(self.processor.tokenizer.eos_token_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return {
            "input_values": wav,
            "input_features": input_features,
            "input_ids": input_ids,
            "mel_mask": mel_mask,
        }


@dataclass
class WhisperVQCollator:
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

    def __call__(self, batch):
        # -> {"input_values": ..., "input_features": ..., "input_ids": ..., "decoder_attention_mask": ...}
        max_values_length = max([x["input_values"].shape[-1] for x in batch])
        max_ids_length = max([x["input_ids"].shape[-1] for x in batch])

        input_values = []
        decoder_attention_mask = []
        decoder_input_ids = []
        input_features = torch.stack([x["input_features"] for x in batch])
        encoder_attention_mask = torch.stack([x["mel_mask"] for x in batch])

        for data in batch:
            values_length = data["input_values"].shape[-1]
            x = torch.nn.functional.pad(
                data["input_values"], (0, max_values_length - values_length)
            )
            input_values.append(x)

            ids_length = data["input_ids"].shape[-1]
            ids = torch.nn.functional.pad(
                data["input_ids"],
                (0, max_ids_length - ids_length),
                value=self.processor.tokenizer.pad_token_id,
            )
            decoder_input_ids.append(ids)

            x = torch.zeros(max_ids_length, dtype=torch.float)
            x[:ids_length] = 1
            decoder_attention_mask.append(x)

        decoder_input_ids = torch.stack(decoder_input_ids)
        decoder_attention_mask = torch.stack(decoder_attention_mask)
        labels = decoder_input_ids.clone()
        labels[decoder_attention_mask == 0] = -100
        labels[:, :4] = -100  # BOS, LANG, TRANSCRIBE, NOTIMESTAMPS

        return {
            "input_values": torch.stack(input_values),
            "input_features": input_features,
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids[:, :-1],
            "decoder_attention_mask": decoder_attention_mask[:, :-1],
            "labels": labels[:, 1:],
        }


if __name__ == "__main__":
    import soundfile as sf
    from torch.utils.data import DataLoader
    from transformers import GenerationConfig

    from speech_lm.models.whisper_vq import WhisperVQ
    from speech_lm.modules.flash_whisper import FlashWhisperForConditionalGeneration

    dataset = WhisperVQDataset("filelists/whisper-vq.test.filelist")
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=WhisperVQCollator()
    )
    # whisper = FlashWhisperForConditionalGeneration.from_pretrained(
    #     "openai/whisper-medium"
    # )
    # whisper.eval()
    our_whisper = WhisperVQ()
    whisper = our_whisper.whisper
    our_whisper.eval()

    state_dict = torch.load(
        "results/whisper-vq/checkpoints/step_10000.ckpt", map_location="cpu"
    )["model"]
    our_whisper.load_state_dict(state_dict, strict=True)
    # whisper.cuda()

    for batch in dataloader:
        # batch = {k: v.cuda() for k, v in batch.items()}
        print({k: v.shape for k, v in batch.items()})

        outputs = whisper.generate(
            inputs=batch["input_features"],
            max_length=448,
            do_sample=False,
        )

        print(outputs, batch["decoder_input_ids"])
        transcriptions = dataset.processor.batch_decode(
            outputs, skip_special_tokens=True
        )

        print(
            transcriptions,
            dataset.processor.batch_decode(batch["labels"], skip_special_tokens=True),
        )
        sf.write("test.wav", batch["input_values"][0].cpu().numpy(), 16000)

        # Calculate loss
        # encoder_outputs = whisper.model.encoder(
        #     batch["input_features"],
        # )
        encoder_outputs = our_whisper.decode(
            our_whisper.encode(
                batch["input_features"],
            )[0]
        )

        decoder_outputs = whisper.generate(
            # decoder_input_ids=batch["decoder_input_ids"],
            # decoder_attention_mask=batch["decoder_attention_mask"],
            # labels=batch["labels"],
            # generation_config=GenerationConfig(
            #     encoder_outputs=(encoder_outputs,)
            # ),
            encoder_outputs,
            max_length=448,
            do_sample=False,
            # forced_decoder_ids=batch["decoder_input_ids"][:, :4]
            forced_decoder_ids=dataset.processor.get_decoder_prompt_ids(
                language="english", task="transcribe"
            ),
        )

        print("Our transcript:", dataset.processor.batch_decode(decoder_outputs))
        break