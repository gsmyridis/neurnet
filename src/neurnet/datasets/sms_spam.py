from pathlib import Path

import mlx.core as mx
import mlx.data as dx
import pandas as pd

from neurnet.arch.llm.types import Tokenizer

from .types import Dataset, MLXDataLoader
from .utils import download_unzip

SMS_SPAM_LABEL_NO_SPAM = "ham"
SMS_SPAM_LABEL_SPAM = "spam"
SMS_SPAM_LABELS = [SMS_SPAM_LABEL_SPAM, SMS_SPAM_LABEL_NO_SPAM]


class SMSSpamCollection(Dataset):
    SOURCE_LINK = (
        "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    )

    _EXTRACTED_DIR = "sms_spam_collection"
    _FILENAME = "sms_spam_collection.tsv"
    _FILENAME_ORIGINAL = "SMSSpamCollection"

    def __init__(
        self,
        root_dir: str | Path,
        tokenizer: Tokenizer,
        max_sequence_len: int,
        shuffle: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.shuffle = shuffle

    def download_path(self) -> Path:
        return self.root_dir / self._EXTRACTED_DIR / self._FILENAME

    def download(self) -> Path:
        target_path = self.download_path()
        if target_path.exists():
            return target_path

        download_path = download_unzip(
            url=self.SOURCE_LINK,
            root_dir=self.root_dir,
            extracted_path=self._EXTRACTED_DIR,
        )
        source_file_path = download_path / self._FILENAME_ORIGINAL
        if not source_file_path.exists():
            raise FileNotFoundError(f"Expected dataset file at {source_file_path}")
        source_file_path.rename(target_path)

        return target_path

    @staticmethod
    def _prepare_dataset(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
        min_value_count = df["Label"].value_counts().min().item()

        spam_subset = df[df["Label"] == SMS_SPAM_LABEL_SPAM].sample(
            min_value_count, random_state=random_state
        )
        no_spam_subset = df[df["Label"] == SMS_SPAM_LABEL_NO_SPAM].sample(
            min_value_count, random_state=random_state
        )

        df_balanced = pd.concat([spam_subset, no_spam_subset])
        df_balanced["Label"] = (
            df_balanced["Label"]
            .map({SMS_SPAM_LABEL_NO_SPAM: 0, SMS_SPAM_LABEL_SPAM: 1})
            .astype("int8")
        )
        return df_balanced

    def to_mlx(
        self,
        prefetch_batches: int,
        prefetch_worker_threads: int,
    ) -> MLXDataLoader:
        """Return an MLX dataloader for next-token training on the SMS text."""

        _ = self.download()
        df = pd.read_csv(
            self.download_path(), sep="\t", header=None, names=["Label", "Text"]
        )

        return SMSSpamMLXDataLoader(
            df=self._prepare_dataset(df, 123),
            max_sequence_len=self.max_sequence_len,
            tokenizer=self.tokenizer,
            shuffle=self.shuffle,
            prefetch_batches=prefetch_batches,
            prefetch_worker_threads=prefetch_worker_threads,
        )


class SMSSpamMLXDataLoader(MLXDataLoader):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Tokenizer,
        max_sequence_len: int | None,
        shuffle: bool,
        prefetch_batches: int,
        prefetch_worker_threads: int,
    ):
        # Tokenize texts
        tokenized_texts = [list(tokenizer.encode(text)) for text in df["Text"]]
        if max_sequence_len is None:
            self._max_sequence_len = max(len(text) for text in tokenized_texts)
        else:
            self._max_sequence_len = max_sequence_len
            tokenized_texts = [seq[: self._max_sequence_len] for seq in tokenized_texts]

        tokenized_texts = [
            tokens
            + [tokenizer.end_of_sequence_token_id()]
            * (self._max_sequence_len - len(tokens))
            for tokens in tokenized_texts
        ]

        # Create buffer and stream
        buffer = dx.buffer_from_vector(
            [
                {"inputs": mx.array(tokens), "targets": mx.array(label)}
                for tokens, label in zip(tokenized_texts, df["Label"])
            ]
        )
        if shuffle:
            buffer = buffer.shuffle()
        self._num_batches = len(buffer)

        self._stream = buffer.to_stream().prefetch(
            prefetch_batches, prefetch_worker_threads
        )

        # Labels
        self._labels = [mx.array(l) for l in df["Label"]]

    def __len__(self) -> int:
        return self._num_batches

    def __next__(self) -> tuple[mx.array, mx.array]:
        next_sample = next(self._stream)
        return mx.array(next_sample["inputs"]), mx.array(next_sample["targets"])

    def stream(self):
        return self._stream
