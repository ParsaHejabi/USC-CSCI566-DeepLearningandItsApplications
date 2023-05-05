# %% [markdown]
# # Importing the libraries

import json
import os
import sys

import requests
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# %%
from transformers import (
    AutoTokenizer,
    RobertaModel,
    ViltConfig,
    ViltModel,
    ViltProcessor,
)

# %%
# Get current working directory
cwd = os.getcwd()
# VQA folder path
vqa_path = os.path.join(cwd, "VQA")
# Add VQA folder to path
sys.path.append(vqa_path)
sys.path.append(os.path.join(vqa_path, "PythonEvaluationTools"))
sys.path.append(os.path.join(vqa_path, "PythonHelperTools"))

from vqaEvaluation import vqaEval
from vqaTools import vqa

# %% [markdown]
# # Defining the device

# %%
device = ""
# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("mps")

print("Using device:", device)

# %% [markdown]
# # Checking the ViLT model

# %%
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vilt_config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vilt_model = ViltModel.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa",
    num_labels=len(vilt_config.id2label),
    id2label=vilt_config.id2label,
    label2id=vilt_config.label2id,
)
vilt_model.to(device)

# %%
# prepare image + question
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "Where the cats are sleeping?"

# %%
image

# %%
# prepare inputs
encoding = vilt_processor(image, text, return_tensors="pt")
encoding = {k: v.to(device) for k, v in encoding.items()}
print(f"Encoding keys: {encoding.keys()}")
print(f"Encoding shape: {encoding['input_ids'].shape}")

# %%
# forward pass
outputs = vilt_model(**encoding)
last_hidden_states = outputs.last_hidden_state
print(f"last_hidden_states shape: {last_hidden_states.shape}")

# %% [markdown]
# # Creating the dataset

# %%
dataset = load_dataset("Multimodal-Fatima/VQAv2_validation")
dataset["validation"][0].keys()
print(f"Length of dataset: {len(dataset['validation'])}")

# %%
SAMPLED_DATASET_SIZE = len(dataset["validation"]) // 10
sampled_dataset_generator = torch.Generator().manual_seed(42)
sampled_dataset_split = torch.utils.data.random_split(
    dataset["validation"],
    [SAMPLED_DATASET_SIZE, len(dataset["validation"]) - SAMPLED_DATASET_SIZE],
    generator=sampled_dataset_generator,
)
sampled_dataset = {
    "train": sampled_dataset_split[1],
    "test": sampled_dataset_split[0],
}
print(f"Train size: {len(sampled_dataset['train'])}")
print(f"Test size: {len(sampled_dataset['test'])}")

# %%
sampled_dataset["train"][0]["image"]


# %%
def get_score(count: int) -> float:
    return min(1.0, count / 3)


# %%
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, split):
        self.dataset = dataset[split]
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question = self.dataset[idx]["question"]
        image = self.dataset[idx]["image"].convert("RGB")
        answers = self.dataset[idx]["answers"]

        answer_counts = {}
        for answer in answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1

        labels = []
        scores = []
        for answer in answer_counts:
            if answer not in list(vilt_config.label2id.keys()):
                continue
            labels.append(vilt_config.label2id[answer])
            scores.append(get_score(answer_counts[answer]))

        encoding = self.processor(
            image, question, padding="max_length", truncation=True, return_tensors="pt"
        )

        for key, value in encoding.items():
            encoding[key] = value.squeeze(0)

        targets = torch.zeros(len(vilt_config.label2id))
        for label, score in zip(labels, scores):
            targets[label] = score

        encoding["labels"] = targets
        encoding["caption"] = self.dataset[idx]["blip_caption"]
        encoding["question_id"] = self.dataset[idx]["question_id"]

        return encoding


# %%
vqa_dataset_train = VQADataset(sampled_dataset, vilt_processor, "train")
vqa_dataset_test = VQADataset(sampled_dataset, vilt_processor, "test")


# %%
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    pixel_values = [item["pixel_values"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    token_type_ids = [item["token_type_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    captions = [item["caption"] for item in batch]
    question_id = [item["question_id"] for item in batch]

    # create padded pixel values and corresponding pixel mask
    encoding = vilt_processor.feature_extractor.pad_and_create_pixel_mask(
        pixel_values, return_tensors="pt"
    )

    # create new batch
    batch = {}
    batch["input_ids"] = torch.stack(input_ids)
    batch["attention_mask"] = torch.stack(attention_mask)
    batch["token_type_ids"] = torch.stack(token_type_ids)
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = torch.stack(labels)
    batch["caption"] = captions
    batch["question_id"] = question_id

    return batch


# %%
train_dataloader = torch.utils.data.DataLoader(
    vqa_dataset_train,
    collate_fn=collate_fn,
    batch_size=64,
    shuffle=True,
    # num_workers=4
)

test_dataloader = torch.utils.data.DataLoader(
    vqa_dataset_test,
    collate_fn=collate_fn,
    batch_size=64,
    shuffle=False,
    # num_workers=4
)

# %%
next(iter(train_dataloader)).keys()

# %%
next(iter(train_dataloader))["pixel_values"].shape

# %% [markdown]
# # RoBERTa model

# %%
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base")
roberta_model.to(device)

# %% [markdown]
# # Decoder model


# %%
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.GELU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_size, output_size))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# %%
input_size = 1536  # vilt_pooled_size + roberta_pooled_size
hidden_size = input_size * 2  # or any other suitable size
output_size = 3129  # number of answer classes
num_layers = 3  # or any other suitable number of layers

decoder = Decoder(input_size, hidden_size, output_size, num_layers).to(device)

# %% [markdown]
# # Training Decoder model

# %%
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-5)


# %%
def vilt_to_vqa_annotation(vilt_data):
    dummy_annotate = """{"info": {"description": "This is Balanced Binary Abstract Scenes VQA dataset.", 
      "url": "http://visualqa.org", 
      "version": "1.0", 
      "year": "2017", 
      "contributor": "VQA Team", 
      "date_created": "2017-03-09 14:27:27"}, 
      "license": {"url": "http://creativecommons.org/licenses/by/4.0/", 
      "name": "Creative Commons Attribution 4.0 International License"}, 
      "data_subtype": "val2017"}"""

    annotate_full = json.loads(dummy_annotate)

    dummy_questions = """
    {"info": {"description": "This is v1.0 of the VQA dataset.", 
    "url": "http://visualqa.org", 
    "version": "1.0", 
    "year": 2015, "contributor": 
    "VQA Team", "date_created": "2015-10-02 19:50:36"}, 
    "task_type": "Open-Ended", 
    "data_type": "abstract_v002", 
    "license": {"url": "http://creativecommons.org/licenses/by/4.0/", 
      "name": "Creative Commons Attribution 4.0 International License"}, 
    "data_subtype": "val2015"}
    """

    question_full = json.loads(dummy_questions)

    annotate_list = []

    question_list = []

    for datapoint in vilt_data:
        """
        print('========')
        for key in datapoint.keys():
          print(key)
        print('========')
        """

        # VILT converted VQA keys, so we need to convert them back
        datapoint["image_id"] = datapoint["id_image"]
        datapoint["answers"] = datapoint["answers_original"]
        # Throw away unneeded stuff
        datapoint["image"] = None
        datapoint["LLM_Description_gpt3_downstream_tasks_visual_genome_ViT_L_14"] = None
        datapoint["DETA_detections_deta_swin_large_o365_coco_classes"] = None
        datapoint["DETA_detections_deta_swin_large_o365_clip_ViT_L_14"] = None
        datapoint[
            "DETA_detections_deta_swin_large_o365_clip_ViT_L_14_blip_caption"
        ] = None

        question_item = {
            "question_id": datapoint["question_id"],
            "image_id": datapoint["image_id"],
            "question": datapoint["question"],
        }

        question_list.append(question_item)
        annotate_list.append(datapoint)

    annotate_full["annotations"] = annotate_list
    question_full["questions"] = question_list

    return {"annotations": annotate_full, "questions": question_full}


# %%
PRINT_EVERY = 10
EPOCHS = 15

epoch_tqdm_bar = tqdm(range(EPOCHS), desc="Epoch")
for epoch in epoch_tqdm_bar:
    epoch_tqdm_bar.set_description(f"Epoch {epoch}")
    batch_tqdm_bar = tqdm(train_dataloader, desc="Batch")

    decoder.train()
    total_train_loss = 0
    for i, batch in enumerate(batch_tqdm_bar, 1):
        batch_tqdm_bar.set_description(f"Batch {i}")
        caption = batch.pop("caption")
        labels = batch.pop("labels").to(device)
        question_id = batch.pop("question_id")

        batch = {k: v.to(device) for k, v in batch.items()}

        vilt_output = vilt_model(**batch)
        vilt_pooled_output = vilt_output.pooler_output

        roberta_tokenized_captions = roberta_tokenizer(
            caption, return_tensors="pt", padding=True
        ).to(device)
        roberta_output = roberta_model(**roberta_tokenized_captions)
        roberta_pooled_output = roberta_output.pooler_output

        concatenated_output = torch.cat(
            (vilt_pooled_output, roberta_pooled_output), dim=-1
        )

        logits = decoder(concatenated_output)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        if i % PRINT_EVERY == 0:
            avg_train_loss = total_train_loss / PRINT_EVERY
            print(f"  Batch {i}, Average Training Loss: {avg_train_loss:.4f}")
            total_train_loss = 0

    torch.save(decoder.state_dict(), f"decoder_model_epoch_{epoch}.pth")

    # Evaluation loop for one epoch
    decoder.eval()
    results = []
    answers = []
    with torch.no_grad():
        test_batch_tqdm_bar = tqdm(test_dataloader, desc="Test Batch")
        for batch in test_batch_tqdm_bar:
            test_batch_tqdm_bar.set_description(f"Test Batch {i}")
            caption = batch.pop("caption")
            labels = batch.pop("labels").to(device)
            question_id = batch.pop("question_id")

            batch = {k: v.to(device) for k, v in batch.items()}

            vilt_output = vilt_model(**batch)
            vilt_pooled_output = vilt_output.pooler_output

            roberta_tokenized_captions = roberta_tokenizer(
                caption, return_tensors="pt", padding=True
            ).to(device)
            roberta_output = roberta_model(**roberta_tokenized_captions)
            roberta_pooled_output = roberta_output.pooler_output

            concatenated_output = torch.cat(
                (vilt_pooled_output, roberta_pooled_output), dim=-1
            )

            logits = decoder(concatenated_output)

            for i in range(len(logits)):
                idx = logits[i].argmax(-1).item()
                answer = vilt_model.config.id2label[idx]
                answers.append(answer)

                results.append({"question_id": question_id[i], "answer": answer})

        convert_data = vilt_to_vqa_annotation(sampled_dataset["test"])

        convert_annotate, convert_questions = (
            convert_data["annotations"],
            convert_data["questions"],
        )

        with open(f"./annotate_epoch_{epoch}.json", "w") as annotate_file:
            annotate_file.write(json.dumps(convert_annotate))

        with open(f"./questions_epoch_{epoch}.json", "w") as annotate_file:
            annotate_file.write(json.dumps(convert_questions))

        with open(f"./results_epoch_{epoch}.json", "w") as res_file:
            res_file.write(json.dumps(results))
