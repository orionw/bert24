# Copyright 2024 BERT24 authors
# SPDX-License-Identifier: Apache-2.0

# """Contains GLUE job objects for the simple_glue_trainer."""
import os
import sys
import json
from typing import List, Optional
from multiprocessing import cpu_count
import torch
from typing import Tuple
from pathlib import Path

# Add glue folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from composer import ComposerModel
from composer.core import Callback
from composer.core.evaluator import Evaluator
from composer.loggers import LoggerDestination
from composer.optim import ComposerScheduler, DecoupledAdamW
from torch.optim import Optimizer
from src.evals.data import create_glue_dataset
from src.evals.finetuning_jobs import build_dataloader, ClassificationJob
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from composer.metrics.nlp import MaskedAccuracy
from datasets import load_dataset
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers.data.data_collator import DataCollatorWithPadding

"""
Q: if for each question, we want to create 5 training examples, how is that expressed?

Background: There is a pipeline going JSON->MCQADataset->OtherThing(=Trainer?)-> training event at training time.

Possible answers:

1. 5 items in the JSON which is loaded into MCQADataset
2. 1 item in the JSON, but the MCQADataset processes it to express 5 items via __geitem__
3. 1 item in the JSON, 1 item in the MCQADataset, but some other process (the Trainer?) expresses it as 5 items before trainingt time.

QQ: How does an existing multiple-choice QA dataset already express this?

Wayde believes re typical HF dataset practice:
- each HF Dataset item is a single question
- 


"SQuAD"-style

different possible data layouts for training and inference for MC questions:

regarding example definition for training and inference:

1. "HF AutoModel for MC." every q/a pair is one example.
   - so, every question leads to ~5 examples
   - at inference:
     - the model is a binary classifier
       - (conveniently, this removes the concern that the model assumes that classification indexes have a stable meaning across examples as in a normal classiferi)
     - the feed all pairs to the model at the same time, and then pick the one with the highest true prediction as the correct.
   - at training time:
     - train only on the binary classifier predictions.
   - we call this "pseudo-contrastive" bc the model is not actually predicting based on seeing a true/false pairs as in true contrastive training
2. "SQuAD-like."
   - every example is (q/context/a1..a5, {correct label index})
     - (room for variation in how to format the ultimate prompt which communicates all that. )
   - model is a 5-way classifier, and predicts class 1..5
   - (what bw describes in message of 2024-11-26)
3. Dan Hendrycks style. (mentinoned by Ben Clavié)

variables to explore:

- which of the above "example definiition" styles to use
- how much context to supply the model.
- quality of the bad answers (ensuring they are present in the text, even if they are incorrect)
  - want to be sure that mere text search does not suffice to identify the correct answer
  - needs experimentation with datasynthesis prompt.
- format of prompt for the ModernBERT model itself
- different prompt styles perhaps as generated by LLMs.
- possibly, prompting the model to generate an alphabetical index of the correct answer, rather than a direct classification


regarding concrete data shape:

- should we expect the inputs to be shaped like: (batch_size X multichoice_count X seq_length)
  - bw: does not see why it should be this way.


2 questions to answer:
- [ ] what data layout do we want to use for the raw dataset
  - bw says: current exported format but removing qd_prompt and answer.
- [ ] what data layout do we want to preprocess it into:

Need to define a CustomDataset(Dataset) which:
- implement len getitem
- does the preprocessing of concatenating to produce the prompt (good me done AOT or lazily in getitem)
- do the tokenization
- in getitem, returns a type+shape which matches the expectations of the existing HF Dataloader/collator for MCQA
  - probably means: returns a {"input_ids":..., "labels":...}

"""


def mk_prompt(mcqa_item: dict) -> str:
    question, evidence, options = map(mcqa_item.get, ["question", "context", "options"])
    choices = "\n".join(["- " + opt for opt in options])

    return f"""Please carefully review the following textual Evidence. It contains information relevant to the Question. Then select the correct answer from the Choices.

    ## Evidence:
    {evidence}

    ## Question
    {question}

    ## Choices
    {choices}
"""


class TriviaMCQA(Dataset):
    def __init__(
        self, path_to_json: Path, split: str, pretrained_model_name_or_path: str = "bclavie/olmo_bert_template"
    ):
        super().__init__()
        with open(path_to_json, "r") as f:
            self.items = json.load(f)[split]
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        print(f"TriviaMCQA Dataset created")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        qaitem = self.items[idx]
        prompt = mk_prompt(qaitem)
        label = qaitem["answer_index"]
        # we use "input_ids" (for the tokens in a single example seq) and "labels" (for a single scalar label) because
        # these are the conventional HF key names, defining a de-facto API for interop with HF tools
        return dict(input_ids=torch.tensor(self.tokenizer.encode(prompt)), labels=torch.tensor(label))




# def collate_padmask(xs, pad_token_id: int, max_seq_length: int = 8192):
#     """
#     list of items -> batch
#     - where item:dict(input_ids=.., labels=...)
#     - adds padding per sequence
#     - adds attention mask
#     """
#     seqs = [x["input_ids"] for x in xs]
#     max_len = min(max_seq_length, max([len(x) for x in seqs]))
#     pseqs = [F.pad(seq, (0, max_len - len(seq)), value=pad_token_id) for seq in seqs]
#     batch_labels = torch.vstack([x["labels"] for x in xs])
#     batch_inputs = torch.vstack(pseqs)
#     batch_mask = (batch_inputs != pad_token_id).long()
#     return dict(input_ids=batch_inputs, attention_mask=batch_mask, labels=batch_labels)


# class MultipleChoiceMaskedAccuracy(MaskedAccuracy):
#     def __init__(self, ANSWER_IDS, ignore_index: int = -100, dist_sync_on_step: bool = False):
#         """
#         ANSWER_IDS : dims of [multichoice_count]
#         """
#         super().__init__(ignore_index, dist_sync_on_step)
#         self.ANSWER_IDS = ANSWER_IDS

#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         # predictions is a batch x num_classes tensor, take the argmax to get class indices
#         preds = torch.argmax(preds, dim=-1)
#         assert preds.shape == target.shape

#         # mask out the padded indices
#         mask = (target != self.ignore_index)
#         masked_target = target[mask]
#         masked_preds = preds[mask]

#         self.correct += torch.sum(masked_preds == masked_target)
#         self.total += mask.sum()


# to be aware of DataCollatorForMultipleChoice is a class which implements data collation
# assuming a slightly different tensor format of bs x multichoice_count x seq length.
# the head of the model is written in such a way that it can consume that format or the bs x seq_length format.


class TriviaMCQAJob(ClassificationJob):
    """TriviaMCQA."""

    num_labels = 3

    def __init__(
        self,
        model: ComposerModel,
        tokenizer_name: str,
        job_name: Optional[str] = None,
        seed: int = 42,
        eval_interval: str = "2300ba",
        scheduler: Optional[ComposerScheduler] = None,
        optimizer: Optional[Optimizer] = None,
        max_sequence_length: Optional[int] = 256,
        max_duration: Optional[str] = "1ep",
        batch_size: Optional[int] = 64,
        load_path: Optional[str] = None,
        save_folder: Optional[str] = None,
        loggers: Optional[List[LoggerDestination]] = None,
        callbacks: Optional[List[Callback]] = None,
        precision: Optional[str] = None,
        opt_default_name: str = "decoupled_adamw",
        opt_lr: float = 1.0e-5,
        opt_betas: Tuple[float, float] = (0.9, 0.98),
        opt_eps: float = 1.0e-6,
        opt_weight_decay: float = 1.0e-5,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer_name=tokenizer_name,
            job_name=job_name,
            seed=seed,
            task_name="triviamcqa",
            eval_interval=eval_interval,
            scheduler=scheduler,
            optimizer=optimizer,
            max_sequence_length=max_sequence_length,
            max_duration=max_duration,
            batch_size=batch_size,
            load_path=load_path,
            save_folder=save_folder,
            loggers=loggers,
            callbacks=callbacks,
            precision=precision,
            **kwargs,
        )
        # use prepare_triviaMCQA to generate this file:
        fname_raw = "triviamcqa100.json"

        # grab longcontext dataset
        # (we expect that tokenizer_name will be "bclavie/olmo_bert_template")
        train_ds = TriviaMCQA(path_to_json=fname_raw, split="train", pretrained_model_name_or_path=tokenizer_name)
        val_ds = TriviaMCQA(path_to_json=fname_raw, split="validation", pretrained_model_name_or_path=tokenizer_name)

        dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": min(8, cpu_count() // torch.cuda.device_count()),
            "drop_last": False,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        collator = DataCollatorWithPadding(self.tokenizer, padding="longest", max_length=max_sequence_length)
        self.train_dataloader = build_dataloader(train_ds, collator, **dataloader_kwargs)

        evaluator = Evaluator(
            label="lc_trivia_mcqa",
            dataloader=build_dataloader(val_ds, collate_fn=collator),
            metric_names=["MulticlassAccuracy"],
        )

        self.evaluators = [evaluator]
        print(f"TriviaMCQA.evaluators defined")
