# PreScam

Code and demo data for **PreScam: A Benchmark for Predicting Scam Progression from Early Conversations** (COLM 2026) [[arXiv](https://arxiv.org/abs/2605.12243)].

PreScam contains 11,573 real-world scam conversations across 20 scam categories. Each conversation is structured into three stages, *Initial Contact*, *Engagement*, and *Termination*, and every scammer action is annotated with psychological techniques (PTs). The benchmark defines two tasks:

- **Real-time Termination Prediction**: from a partial conversation, predict whether the scammer's next move enters the termination stage.
- **Scammer Action Prediction**: from the observed prefix, forecast the scammer's subsequent actions.

## Data

`data/demo.json` contains 100 example instances. The full dataset is available on request; see [`data/README.md`](data/README.md).

## Repository structure

```
data/                     demo data, PT taxonomy, data access instructions
termination_prediction/   real-time termination prediction (classical, neural, and LLM baselines)
action_prediction/        boundary classifier + zero-shot LLM action prediction and judge-based evaluation
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
```

## Real-time Termination Prediction

```bash
cd termination_prediction
python rt_detect/realtime_detection.py --config rt_detect/config.yaml
python rt_detect/realtime_detection.py --methods tfidf bert          # subset of methods
python rt_detect/run_k_ablation.py --k-values 1 2 3                   # positive-window ablation
```

Methods: `position_only`, `tfidf`, `mlp_tfidf`, `mlp_embed`, `lstm`, `transformer`, `hierarchical`, `bert`, `llm`. Choose the LLM with `llm_model` in `config.yaml`. LLMs are called through OpenRouter. The scripts report AUC, AUPR, and AT@FPR<sub>10%</sub>.

## Scammer Action Prediction

Requires the full train/val/test splits and the turn-level labels (see `data/README.md`).

```bash
cd action_prediction
python train.py                                                       # train the RoBERTa boundary classifier
python baselines/run_baselines.py --model gpt-4o-mini --no-limit-turns   # Unlimited
python baselines/run_baselines.py --model gpt-4o-mini --limit-turns      # Limited
bash baselines/run_all.sh                                             # all models × both settings
```

The scripts report Action HitRate, PT HitRate, and Precision using a GPT-4o-mini judge, plus BERTScore and ROUGE-L.

## Citation

```bibtex
@inproceedings{sun2026prescam,
  title     = {PreScam: A Benchmark for Predicting Scam Progression from Early Conversations},
  author    = {Sun, Weixiang and Ma, Shang and Li, Yiyang and Ma, Tianyi and Wang, Zehong and Nelson, Colby and Xiao, Xusheng and Ye, Yanfang},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## Intended use

PreScam is released for research and defensive anti-scam applications only. Do not use the data or code to author, optimize, or rehearse scam content, or to identify individuals mentioned in the reports.
