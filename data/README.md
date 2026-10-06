# PreScam Data

This folder contains a **demo subset** of PreScam (`demo.json`, 100 instances) and the psychological-technique (PT) taxonomy (`pt_definitions.json`).

The full benchmark (11,573 structured scam instances across 20 scam categories) is **available on request**. It is not publicly downloadable.

## Why the full dataset is not public

- **Real victim reports.** Every instance comes from a real report submitted to BBB Scam Tracker. Each instance includes the victim's original narrative, which can contain personal details, locations, or descriptions of financial loss. We have not manually de-identified every one of these reports.
- **Dual-use risk.** PreScam breaks real scams down step by step, labeling each scammer action with the psychological technique it uses. That structure is useful for detection and for warning users, but it could also be misused to write, optimize, or rehearse scam scripts.
- **Source terms.** The underlying reports come from a third-party platform. Controlled access lets us make sure the data is used for non-commercial research that is consistent with those terms.

## How to request access

Email both **wsun4@nd.edu** and **sma5@nd.edu**.

In your email, include:

1. Your name, affiliation, and position (if you are a student, also include your advisor's name and email)
2. A short description of your intended research use
3. Confirmation that you agree to the terms below

### Terms of use

By requesting the data, you agree to:

- use it only for non-commercial research or for defensive/educational anti-scam purposes;
- not use it to author, optimize, or rehearse scam or manipulation content, or to build systems whose purpose is deception;
- not attempt to identify, contact, or profile any victim or other individual mentioned in the reports;
- not redistribute the data, in whole or in part, to anyone else; others should request access themselves;
- cite the PreScam paper in any work that uses the data.

## Demo format

`demo.json` is a list of instances in exactly the same format as the full dataset:

| Field | Description |
|---|---|
| `scam_id` | Report identifier |
| `scam_type` | One of 20 scam categories |
| `description` | Original victim narrative |
| `initial_contact` | Summary of how the scam begins (*Initial Contact* stage) |
| `engagement` | List of turns; each has `scammer_action`, `scammer_action_verbatim`, `victim_action`, `victim_action_verbatim`, and `PTs` |
| `outcome` | Summary of the final stage (*Termination* in the paper) |
| `num_engagement_rounds` | Number of engagement turns |
| `scammed` | 1 if the scammer succeeded, else 0 |
| `scammed_reason` | Explanation of the `scammed` label |

PT labels use the 9-label taxonomy in `pt_definitions.json`.

## Full dataset layout

The full release is laid out like this; place the files in this folder to run the code:

```
data/
  train.json  val.json  test.json                                   # 8:1:1 split, same format as demo.json
  train_llm_labels.jsonl  val_llm_labels.jsonl  test_llm_labels.jsonl  # turn-level boundary labels
```

To run real-time termination prediction on the full data, set `input` in `termination_prediction/rt_detect/config.yaml` to the full data file.
