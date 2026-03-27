#!/usr/bin/env python3
"""
Step 2 (LLM version): Extract Aortic Dilation Labels using LLM

Filters reports to only studies with DICOM files on disk (~647),
then uses a local LLM (Qwen via vLLM) to extract aortic dilation info.

Flow:
  1. Load accession index (from Step 1) to know which studies have DICOMs
  2. Load metadata CSV and keep ONLY reports whose acc_num is in the index
  3. Run LLM on those ~647 reports (takes minutes, not hours)
  4. Save labeled CSV for training

Usage:
  python scripts/02_extract_labels_llm.py --config configs/default.yaml
  python scripts/02_extract_labels_llm.py --config configs/default.yaml --model Qwen/Qwen2.5-7B-Instruct
"""

import os
import sys
import json
import argparse
import logging
import re
import time

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Prompt template                                                     #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are a cardiology expert assistant. Your task is to extract 
aortic dilation information from echocardiogram reports.

Respond ONLY with a JSON object, no other text. The JSON must have these fields:
{
  "aortic_dilation": "yes" or "no",
  "severity": "normal", "mild", "moderate", or "severe",
  "aortic_root_diameter_cm": number or null,
  "ascending_aorta_diameter_cm": number or null,
  "evidence": "relevant quote or summary from the report",
  "confidence": "high", "medium", or "low"
}

Guidelines:
- "aortic_dilation" = "yes" if the aortic root OR ascending aorta is dilated/enlarged/aneurysmal.
- Aortic root >= 4.0 cm is generally considered dilated.
- Ascending aorta >= 3.8 cm is generally considered dilated.
- If the report says "normal aortic root" or "aortic root is normal in size", that is "no".
- If the report does not mention the aorta at all, set "aortic_dilation" to "no", 
  confidence to "low", and evidence to "no mention of aorta in report".
- If the report mentions prior dilation that has resolved, that is "no".
- Extract numeric measurements if present (convert mm to cm).
- For severity: mild = 4.0-4.4 cm, moderate = 4.5-5.4 cm, severe >= 5.5 cm (aortic root).
- Set confidence to "high" if there is a clear statement or measurement,
  "medium" if the language is ambiguous, "low" if you are guessing."""

USER_PROMPT_TEMPLATE = """Extract aortic dilation information from this echocardiogram report:

---
{report_text}
---

Respond with JSON only."""


# ------------------------------------------------------------------ #
#  Data loading                                                        #
# ------------------------------------------------------------------ #

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_accession_index(index_path: str) -> dict:
    """Load the accession index built by Step 1."""
    if not os.path.exists(index_path):
        logger.error(f"Accession index not found: {index_path}")
        logger.error("Run 01_explore_metadata.py first.")
        sys.exit(1)

    with open(index_path) as f:
        index = json.load(f)

    logger.info(f"Loaded accession index: {len(index)} studies with DICOMs on disk")
    return index


def load_and_filter_reports(metadata_dir: str, valid_accessions: set) -> pd.DataFrame:
    """
    Load echo reports from the metadata CSV, but ONLY keep reports
    whose accession number has matching DICOM files on disk.

    This reduces 634K reports down to ~647 — the ones we can actually
    use for training.
    """
    procedures_file = os.path.join(metadata_dir, "dbo.derived_cardiology_echo_procedures.csv")

    if not os.path.exists(procedures_file):
        logger.error(f"Procedures file not found: {procedures_file}")
        return pd.DataFrame(columns=["accession", "report_text"])

    file_size_gb = os.path.getsize(procedures_file) / (1024**3)
    logger.info(f"Reading {procedures_file} ({file_size_gb:.1f} GB)")
    logger.info(f"Filtering to {len(valid_accessions)} accessions with DICOMs on disk")

    use_cols = ["acc_num", "narrative", "impression"]
    all_chunks = []
    chunk_size = 50000

    try:
        for i, chunk in enumerate(pd.read_csv(
            procedures_file,
            usecols=use_cols,
            chunksize=chunk_size,
            dtype=str,
            on_bad_lines="skip",
        )):
            # Drop rows without accession or report text
            chunk = chunk.dropna(subset=["acc_num"])
            chunk = chunk.dropna(subset=["narrative", "impression"], how="all")

            # FILTER: only keep rows with matching DICOMs
            chunk = chunk[chunk["acc_num"].isin(valid_accessions)]

            if len(chunk) > 0:
                all_chunks.append(chunk)

            if (i + 1) % 10 == 0:
                total_so_far = sum(len(c) for c in all_chunks)
                logger.info(f"  Scanned {(i+1) * chunk_size:,} rows, matched {total_so_far} so far...")

    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        if not all_chunks:
            return pd.DataFrame(columns=["accession", "report_text"])

    if not all_chunks:
        logger.warning("No reports matched any accession in the DICOM index!")
        return pd.DataFrame(columns=["accession", "report_text"])

    df = pd.concat(all_chunks, ignore_index=True)
    logger.info(f"Matched reports: {len(df)}")

    # Combine narrative + impression
    df["narrative"] = df["narrative"].fillna("")
    df["impression"] = df["impression"].fillna("")
    df["report_text"] = df["narrative"] + "\n\n" + df["impression"]

    # Clean up whitespace
    df["report_text"] = df["report_text"].str.replace(r"\r\n", " ", regex=True)
    df["report_text"] = df["report_text"].str.replace(r"\s+", " ", regex=True)
    df["report_text"] = df["report_text"].str.strip()

    df = df.rename(columns={"acc_num": "accession"})
    df = df.drop_duplicates(subset=["accession"])

    logger.info(f"Unique matched studies: {len(df)}")

    # Report accessions in DICOM index that had no matching report
    matched_acc = set(df["accession"].astype(str))
    unmatched = valid_accessions - matched_acc
    if unmatched:
        logger.warning(
            f"{len(unmatched)} accessions have DICOMs but NO matching report in metadata. "
            f"These studies cannot be labeled."
        )

    return df[["accession", "report_text"]]


# ------------------------------------------------------------------ #
#  LLM inference with vLLM                                             #
# ------------------------------------------------------------------ #

def build_prompts(reports_df: pd.DataFrame) -> list[list[dict]]:
    """Build chat-style prompts for each report."""
    prompts = []
    for _, row in reports_df.iterrows():
        report_text = row["report_text"]
        words = report_text.split()
        if len(words) > 3000:
            report_text = " ".join(words[:3000]) + "\n[... report truncated ...]"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(report_text=report_text)},
        ]
        prompts.append(messages)

    return prompts

def run_vllm_inference(
    prompts: list[list[dict]],
    accessions: list[str],
    model_name: str,
    checkpoint_path: str,
    max_tokens: int = 300,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    batch_size: int = 100,
) -> list[str]:
    """
    Run batched inference using vLLM with checkpointing.

    Processes prompts in batches, saving a checkpoint after each batch.
    On resume, skips already-processed accessions.
    """
    from vllm import LLM, SamplingParams

    # Resume from checkpoint if exists
    done_accessions = set()
    done_responses = {}
    if os.path.exists(checkpoint_path):
        ckpt_df = pd.read_csv(checkpoint_path)
        done_accessions = set(ckpt_df["accession"].astype(str).tolist())
        done_responses = dict(zip(ckpt_df["accession"].astype(str), ckpt_df["raw_response"]))
        logger.info(f"Resuming from checkpoint: {len(done_accessions)} already processed")

    # Filter out already-done prompts
    remaining_indices = [i for i, acc in enumerate(accessions) if str(acc) not in done_accessions]
    logger.info(f"Remaining to process: {len(remaining_indices)}")

    if len(remaining_indices) == 0:
        logger.info("All prompts already processed!")
        return [done_responses.get(str(acc), "") for acc in accessions]

    # Load model
    logger.info(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Process in batches
    total_batches = (len(remaining_indices) + batch_size - 1) // batch_size
    logger.info(f"Processing in {total_batches} batches of {batch_size}")
    start_time = time.time()

    for batch_num in range(total_batches):
        batch_idx_start = batch_num * batch_size
        batch_idx_end = min(batch_idx_start + batch_size, len(remaining_indices))
        batch_indices = remaining_indices[batch_idx_start:batch_idx_end]

        batch_prompts = [prompts[i] for i in batch_indices]
        batch_accessions = [accessions[i] for i in batch_indices]

        try:
            outputs = llm.chat(batch_prompts, sampling_params=sampling_params)
            for i, output in enumerate(outputs):
                resp = output.outputs[0].text.strip()
                done_responses[str(batch_accessions[i])] = resp
                done_accessions.add(str(batch_accessions[i]))
        except Exception as e:
            logger.error(f"Error in batch {batch_num}: {e}")
            for acc in batch_accessions:
                done_responses[str(acc)] = f'{{"error": "{str(e)}"}}'
                done_accessions.add(str(acc))

        # Save checkpoint
        ckpt_rows = [{"accession": acc, "raw_response": resp} for acc, resp in done_responses.items()]
        pd.DataFrame(ckpt_rows).to_csv(checkpoint_path, index=False)

        elapsed = time.time() - start_time
        done_this_run = batch_idx_end
        rate = done_this_run / elapsed if elapsed > 0 else 0
        logger.info(
            f"Batch {batch_num + 1}/{total_batches} | "
            f"Done: {len(done_accessions)}/{len(accessions)} | "
            f"Rate: {rate:.1f}/s | "
            f"Elapsed: {elapsed:.0f}s"
        )

    logger.info(f"Inference complete in {time.time() - start_time:.1f}s")
    return [done_responses.get(str(acc), "") for acc in accessions]

# ------------------------------------------------------------------ #
#  Parse LLM responses                                                 #
# ------------------------------------------------------------------ #

def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM's JSON response into a structured dict."""
    defaults = {
        "aortic_dilation": "no",
        "severity": "normal",
        "aortic_root_diameter_cm": None,
        "ascending_aorta_diameter_cm": None,
        "evidence": "parse_error",
        "confidence": "low",
    }

    try:
        text = response_text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        parsed = json.loads(text)

        result = {
            "aortic_dilation": str(parsed.get("aortic_dilation", "no")).lower().strip(),
            "severity": str(parsed.get("severity", "normal")).lower().strip(),
            "aortic_root_diameter_cm": parsed.get("aortic_root_diameter_cm"),
            "ascending_aorta_diameter_cm": parsed.get("ascending_aorta_diameter_cm"),
            "evidence": str(parsed.get("evidence", "")),
            "confidence": str(parsed.get("confidence", "low")).lower().strip(),
        }

        if result["aortic_dilation"] not in ("yes", "no"):
            result["aortic_dilation"] = "no"

        valid_severities = ("normal", "mild", "moderate", "severe")
        if result["severity"] not in valid_severities:
            result["severity"] = "normal"

        if result["confidence"] not in ("high", "medium", "low"):
            result["confidence"] = "low"

        return result

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse LLM response: {e}")
        logger.warning(f"Raw response: {response_text[:200]}")
        return defaults


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Extract aortic dilation labels via LLM")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--output-suffix", default="_llm",
                        help="Suffix added to output filename")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    results_dir = os.path.dirname(data_cfg["labels_cache"])

    # 1. Load accession index to know which studies have DICOMs
    logger.info("=" * 60)
    logger.info("STEP 1: Loading accession index")
    logger.info("=" * 60)
    accession_index_path = os.path.join(results_dir, "accession_index.json")
    acc_index = load_accession_index(accession_index_path)
    valid_accessions = set(acc_index.keys())

    # 2. Load reports — filtered to only studies with DICOMs
    logger.info("=" * 60)
    logger.info("STEP 2: Loading and filtering echo reports")
    logger.info("=" * 60)
    reports_df = load_and_filter_reports(data_cfg["jhu_metadata_dir"], valid_accessions)

    if reports_df.empty:
        logger.error("No matching reports found. Exiting.")
        sys.exit(1)

    # 3. Build prompts and run LLM
    logger.info("=" * 60)
    logger.info("STEP 3: Running LLM inference via vLLM")
    logger.info("=" * 60)
    prompts = build_prompts(reports_df)
    logger.info(f"Built {len(prompts)} prompts")

    checkpoint_path = os.path.join(results_dir, "llm_labels_checkpoint.csv")
    responses = run_vllm_inference(
        prompts=prompts,
        accessions=reports_df["accession"].tolist(),
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        tensor_parallel_size=args.tensor_parallel,
    )

    # 4. Parse responses
    logger.info("=" * 60)
    logger.info("STEP 4: Parsing LLM responses")
    logger.info("=" * 60)
    parsed_results = [parse_llm_response(r) for r in responses]

    n_failures = sum(1 for r in parsed_results if r["evidence"] == "parse_error")
    logger.info(f"Parse failures: {n_failures}/{len(parsed_results)}")

    # 5. Build output dataframe
    rows = []
    for i, (acc, parsed) in enumerate(zip(reports_df["accession"].tolist(), parsed_results)):
        label = 1 if parsed["aortic_dilation"] == "yes" else 0
        diameter = parsed.get("aortic_root_diameter_cm")
        if diameter is None:
            diameter = parsed.get("ascending_aorta_diameter_cm")

        # Link to DICOM paths
        dicom_paths = acc_index.get(str(acc), [])

        rows.append({
            "accession": acc,
            "label": label,
            "confidence": parsed["confidence"],
            "evidence": parsed["evidence"],
            "severity": parsed["severity"] if label == 1 else None,
            "diameter_cm": diameter,
            "dicom_path": dicom_paths[0] if dicom_paths else None,
            "num_dicom_files": len(dicom_paths),
            "raw_response": responses[i],
        })

    results_df = pd.DataFrame(rows)

    # 6. Save results
    base_path = data_cfg["labels_cache"]
    name, ext = os.path.splitext(base_path)
    output_path = f"{name}{args.output_suffix}{ext}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Labels saved to {output_path}")

    # Also save raw responses for debugging
    raw_output_path = os.path.join(results_dir, "llm_raw_responses.json")
    with open(raw_output_path, "w") as f:
        json.dump(
            [{"accession": acc, "response": resp, "parsed": parsed}
             for acc, resp, parsed in zip(
                 reports_df["accession"].tolist(), responses, parsed_results
             )],
            f, indent=2,
        )
    logger.info(f"Raw LLM responses saved to {raw_output_path}")

    # 7. Summary
    logger.info("\n" + "=" * 60)
    logger.info("LABEL EXTRACTION SUMMARY (LLM)")
    logger.info("=" * 60)
    logger.info(f"Total studies with DICOMs: {len(valid_accessions)}")
    logger.info(f"Studies with matching reports: {len(results_df)}")
    logger.info(f"Studies with DICOMs but no report: {len(valid_accessions) - len(results_df)}")
    logger.info(f"\nLabel distribution:")
    logger.info(f"{results_df['label'].value_counts().to_string()}")
    logger.info(f"\nConfidence distribution:")
    logger.info(f"{results_df['confidence'].value_counts().to_string()}")

    pos = results_df[results_df["label"] == 1]
    if len(pos) > 0:
        logger.info(f"\nSeverity distribution (among positives):")
        logger.info(f"{pos['severity'].value_counts().to_string()}")

    logger.info(f"\nDICOM files per study (stats):")
    logger.info(f"  Mean: {results_df['num_dicom_files'].mean():.0f}")
    logger.info(f"  Min:  {results_df['num_dicom_files'].min()}")
    logger.info(f"  Max:  {results_df['num_dicom_files'].max()}")

    # Show examples
    logger.info("\n--- Sample POSITIVE cases ---")
    for _, row in results_df[results_df["label"] == 1].head(5).iterrows():
        logger.info(f"  Accession: {row['accession']}")
        logger.info(f"  Evidence:  {row.get('evidence', 'N/A')}")
        logger.info(f"  Severity:  {row.get('severity', 'N/A')}")
        logger.info(f"  Diameter:  {row.get('diameter_cm', 'N/A')}")
        logger.info(f"  # DICOMs:  {row.get('num_dicom_files', 'N/A')}")
        logger.info("")

    logger.info("--- Sample NEGATIVE cases ---")
    for _, row in results_df[(results_df["label"] == 0) & (results_df["confidence"] == "high")].head(3).iterrows():
        logger.info(f"  Accession: {row['accession']}")
        logger.info(f"  Evidence:  {row.get('evidence', 'N/A')}")
        logger.info("")

    logger.info("Done!")
    logger.info(f"To use for training, update labels_cache in default.yaml to: {output_path}")


if __name__ == "__main__":
    main()