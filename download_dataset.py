from datasets import load_dataset
import os

# datasets = [
#     "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
#     "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
#     "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
# ]

output_dir = "./datasets/longbench/data"
os.makedirs(output_dir, exist_ok=True)

# for dataset in datasets:
#     print(f"Downloading: {dataset}")
#     data = load_dataset("THUDM/LongBench", dataset, split="test")
#     out_path = os.path.join(output_dir, f"{dataset}.jsonl")
#     data.to_json(out_path)
#     print(f"Saved to: {out_path}")

e_datasets = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
]

for dataset in e_datasets:
    print(f"Downloading: {dataset}_e")
    data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
    out_path = os.path.join(output_dir, f"{dataset}_e.jsonl")
    data.to_json(out_path)
    print(f"Saved to: {out_path}")