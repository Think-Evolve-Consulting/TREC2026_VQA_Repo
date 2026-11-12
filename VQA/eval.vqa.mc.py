"""
NIST-developed software is provided by NIST as a public service. You may use, copy, and distribute copies of the software in any medium, provided that you
keep intact this entire notice. You may improve, modify, and create derivative works of the software or any portion of the software, and you may copy and
distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of 
any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software. 

NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT, OR ARISING BY OPERATION OF LAW, 
INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, AND DATA ACCURACY. NIST 
NEITHR REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES 
NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY,
RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, 
including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, 
and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury
or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
"""

import pandas as pd
import os
import argparse

def evaluate_runs(ground_truth_file, run_file):
    """Evaluates Top-1 Accuracy and MRR for a single submitted run and generates a diagnostic file."""
    # Load ground truth
    gt_df = pd.read_csv(ground_truth_file, header=None, names=["QueryID", "VideoID", "Question","CorrectAnswer"], dtype=str)
    gt_dict = dict(zip(gt_df["QueryID"], gt_df["CorrectAnswer"]))

    # Load submitted run
    run_df = pd.read_csv(run_file, header=None, names=["QueryID", "VideoID", "Rank", "Response"], dtype=str)
    run_df = run_df.astype(str)
    
    # Ensure the run is sorted by Rank
    run_df = run_df.sort_values(by=["QueryID", "Rank"], key=lambda col: col.astype(int)).reset_index(drop=True)

    # Convert runs into a dictionary (query -> ranked responses)
    run_dict = {}
    for _, row in run_df.iterrows():
        qid = row["QueryID"]
        if qid not in run_dict:
            run_dict[qid] = []
        run_dict[qid].append(row["Response"])  # Maintain ranked order

    # Compute Top-1 Accuracy, MRR, and store per-query details
    top1_correct = 0
    mrr_total = 0
    total_queries = len(gt_dict)
    diagnostic_data = []

    #### skip these queries because their videos were not available for download (18 videos in 2025)
    SKIP = {18,27,62,222,448,507,584,629,903,1007,1159,1260,1330,1381,1554,1691,1755,1822} 

    for qid, correct_answer in gt_dict.items():

        if str(qid).strip().isdigit() and int(qid) in SKIP:
            continue #skip this iteration

        ranked_responses = run_dict.get(qid, [])  # Get responses, default to empty list
        top1_score = 1 if ranked_responses and ranked_responses[0] == correct_answer else 0
        top1_correct += top1_score

        # Find the rank of the correct answer
        try:
            rank = ranked_responses.index(correct_answer) + 1  # Convert zero-based index to rank
            mrr_score = round(1 / rank, 3)  # Round to 3 decimal places
        except ValueError:
            rank = -1  # Not found
            mrr_score = 0.000  # If the correct answer is missing, score is 0

        mrr_total += mrr_score

        # Store diagnostic data per query
        diagnostic_data.append([
            qid, correct_answer, top1_score, mrr_score, rank, str(ranked_responses)
        ])

    # Compute final scores
    top1_accuracy = round(top1_correct / total_queries, 3)
    mrr_score = round(mrr_total / total_queries, 3)

    # Save diagnostic file
    diagnostic_df = pd.DataFrame(
        diagnostic_data,
        columns=["QueryID", "CorrectAnswer", "Top1Correct", "MRRScore", "Rank", "SubmittedResponses"]
    )
    diagnostic_file_path = run_file.replace(".csv", "_diagnostic.csv")
    diagnostic_df.to_csv(diagnostic_file_path, index=False, float_format="%.3f")

    print(f"Diagnostic file saved: {diagnostic_file_path}")

    return top1_accuracy, mrr_score, diagnostic_file_path


def process_folder(ground_truth_file, run_folder):
    """Processes all run files in a given folder and evaluates them."""
    results = []

    # Iterate over all CSV files in the run folder
    for file in os.listdir(run_folder):
        if file.endswith(".csv"):
            run_file_path = os.path.join(run_folder, file)
            top1_accuracy, mrr_score, diagnostic_path = evaluate_runs(ground_truth_file, run_file_path)
            results.append([file, top1_accuracy, mrr_score, diagnostic_path])
            print(f"Processed {file}: Top-1 Accuracy = {top1_accuracy:.3f}, MRR = {mrr_score:.3f}")

    # Save results to a CSV file
    results_df = pd.DataFrame(results, columns=["RunFile", "Top1Accuracy", "MRR", "DiagnosticFile"])
    results_csv_path = os.path.join(run_folder, "evaluation_results.csv")
    results_df.to_csv(results_csv_path, index=False, float_format="%.3f")

    print(f"\nResults saved to: {results_csv_path}")


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Evaluate runs for Top-1 Accuracy and MRR, generating diagnostic reports")
    parser.add_argument("ground_truth", help="Path to the ground truth CSV file")
    parser.add_argument("run_folder", help="Path to the folder containing run files")

    args = parser.parse_args()

    # Run evaluation
    process_folder(args.ground_truth, args.run_folder)
