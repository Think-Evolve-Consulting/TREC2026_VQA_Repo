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
import argparse
import os
import pandas as pd
import numpy as np
import nltk
import subprocess
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from bert_score import score
from sklearn.metrics import ndcg_score

# Ensure NLTK resources are available
nltk.download('wordnet')

# Normalize text: strip spaces, convert to lowercase
def normalize_text(text):
    #print("text =",text)
    return ' '.join(text.strip().lower().split())  # Removes extra spaces and standardizes text

def ndcg_from_gains(gains, k=None):
    """
    gains: list/array of similarity scores (e.g., METEOR or BERT F1) 
           aligned with your current ranking order (best first).
    k:     cutoff; None means use all items.
    """
    gains = np.asarray(gains, dtype=float)
    m = gains.size
    if m == 0:
        return 0.0
    if m == 1:
        return 1.0
    # Build a strictly decreasing score vector to encode the current order.
    sys_scores = np.arange(m, 0, -1, dtype=float)  # m, m-1, ..., 1
    return float(ndcg_score(gains.reshape(1, -1), sys_scores.reshape(1, -1), k=k))

def evaluate_run(ground_truth_file, run_file):
    """Evaluates METEOR, BERTScore, and NDCG using METEOR/BERTScore as relevance scores for a single run."""
    
    # Load ground truth
    gt_df = pd.read_csv(ground_truth_file, header=None, names=["QueryID", "VideoID", "Question","CorrectAnswer"], dtype=str)

    # Ensure all three columns are strings; replace any None/NaN with ""
    gt_df = gt_df.fillna("").astype({"QueryID": str, "VideoID": str, "CorrectAnswer": str})

    # Optionally, warn about rows with missing essential fields
    bad = gt_df[(gt_df["QueryID"] == "") | (gt_df["VideoID"] == "") | (gt_df["CorrectAnswer"] == "")]
    if not bad.empty:
        print(f"Warning: {len(bad)} GT rows missing QueryID or VideoID. Example:\n", bad.head(5))

    gt_dict = {qid: normalize_text(answer) for qid, _, _,answer in gt_df.itertuples(index=False)}

    print("done loading GT")

    # Load submitted run
    print("loading run - ",run_file)
    run_df = pd.read_csv(run_file, header=None, names=["QueryID", "VideoID", "Rank", "Answer", "time"], dtype=str)
    run_df = run_df.astype(str)
    run_df = run_df.sort_values(by=["QueryID", "Rank"], key=lambda col: col.astype(int)).reset_index(drop=True) #sort numerically

    # Ensure all three columns are strings; replace any None/NaN with ""
    run_df = run_df.fillna("").astype({"QueryID": str, "VideoID": str, "Answer": str})
    run_df["QueryID"] = run_df["QueryID"].astype(str).str.strip()

    # Prepare lists for metric calculations
    diagnostic_data = []
    per_response_data = []
    meteor_scores = []
    bert_scores = []
    sts_scores = []
    ndcg_meteor_scores = []
    ndcg_bert_scores = []

    #### skip these queries because their videos were not available for download (18 videos in 2025)
    SKIP = {18,27,62,222,448,507,584,629,903,1007,1159,1260,1330,1381,1554,1691,1755,1822} 

    for qid ,correct_answer in gt_dict.items():
        
        if str(qid).strip().isdigit() and int(qid) in SKIP:
            continue #skip this iteration

        responses = run_df[run_df["QueryID"] == str(qid).strip()]["Answer"].tolist()
        ranks = run_df[run_df["QueryID"] == str(qid).strip()]["Rank"].tolist()
        normalized_responses = [normalize_text(response) for response in responses]  # Normalize responses
        
        if normalized_responses:

            #call STS metric
            sts_relevance = [os.popen("perl calc.sts.pl -sen \""+correct_answer+"\" -sen \""+response+"\"").read() for response in normalized_responses]
            sts_relevance = np.array([np.nan if str(x).strip()=="" else float(x) for x in sts_relevance], dtype=float)

            # Compute METEOR scores with tokenized input
            meteor_relevance = [meteor_score([word_tokenize(correct_answer)], word_tokenize(response)) 
                                for response in normalized_responses]

            # Compute BERTScore F1 for each ranked response
            _, _, F1 = score(normalized_responses, [correct_answer] * len(normalized_responses), lang="en")
            bert_relevance = F1.tolist()  # Convert tensor to list

            # Compute NDCG using METEOR as gain (aligned with current ranked order)
            ndcg_meteor = ndcg_from_gains(meteor_relevance)
            ndcg_meteor_scores.append(ndcg_meteor)

            # Compute NDCG using BERTScore F1 as gain (aligned with current ranked order)
            ndcg_bert = ndcg_from_gains(bert_relevance)
            ndcg_bert_scores.append(ndcg_bert)

            # Use average score per query
            meteor_scores.append(np.mean(meteor_relevance))  
            bert_scores.append(np.mean(bert_relevance))
            sts_scores.append(float(np.nanmean(sts_relevance)))

            # Store per-response metrics
            for rank, response, meteor, bert, sts in zip(ranks, normalized_responses, meteor_relevance, bert_relevance, sts_relevance):
                per_response_data.append([
                    qid, rank, response, correct_answer, round(meteor, 3), round(bert, 3), round(sts,3),
                    round(ndcg_meteor, 3), round(ndcg_bert, 3)
                ])
        else:
            # If no response, assign scores of 0
            meteor_scores.append(0)
            bert_scores.append(0)
            sts_scores.append(0)
            ndcg_meteor_scores.append(0)
            ndcg_bert_scores.append(0)

        # Store diagnostic data for this query
        diagnostic_data.append([
            qid, correct_answer, round(meteor_scores[-1], 3), round(bert_scores[-1], 3), round(sts_scores[-1], 3),
            round(ndcg_meteor_scores[-1], 3), round(ndcg_bert_scores[-1], 3), ', '.join(responses)
        ])

    # Compute overall scores
    avg_meteor = round(np.mean(meteor_scores), 3)
    avg_bert = round(np.mean(bert_scores), 3)
    avg_sts = round(float(np.nanmean(np.array(sts_scores, dtype=float))), 3)
    avg_ndcg_meteor = round(np.mean(ndcg_meteor_scores), 3)
    avg_ndcg_bert = round(np.mean(ndcg_bert_scores), 3)

    # Save diagnostic file
    diagnostic_df = pd.DataFrame(
        diagnostic_data,
        columns=["QueryID", "CorrectAnswer", "METEOR", "BERTScore", "STSscore", "NDCG_METEOR", "NDCG_BERTScore", "Responses"]
    )
    diagnostic_file_path = run_file.replace(".csv", "_diagnostic.csv")
    diagnostic_df.to_csv(diagnostic_file_path, index=False, float_format="%.3f")

    # Save per-response file
    per_response_df = pd.DataFrame(
        per_response_data,
        columns=["QueryID", "Rank", "SubmittedAnswer", "CorrectAnswer", "METEOR", "BERTScore", "STSscore", "NDCG_METEOR", "NDCG_BERTScore"]
    )
    per_response_file_path = run_file.replace(".csv", "_per_response.csv")
    per_response_df.to_csv(per_response_file_path, index=False, float_format="%.3f")

    print(f"Diagnostic file saved: {diagnostic_file_path}")
    print(f"Per-response file saved: {per_response_file_path}")

    return os.path.basename(run_file), avg_meteor, avg_bert, avg_sts, avg_ndcg_meteor, avg_ndcg_bert, diagnostic_file_path, per_response_file_path


def process_folder(ground_truth_file, run_folder):
    """Processes all run files in a given folder and evaluates them."""
    results = []

    for file in os.listdir(run_folder):
        if file.endswith(".csv"):
            run_file_path = os.path.join(run_folder, file)
            run_name, avg_meteor, avg_bert, avg_sts, avg_ndcg_meteor, avg_ndcg_bert, diagnostic_path, per_response_path = evaluate_run(ground_truth_file, run_file_path)
            results.append([run_name, avg_meteor, avg_bert, avg_sts, avg_ndcg_meteor, avg_ndcg_bert, diagnostic_path, per_response_path])
            print(f"Processed {run_name}: METEOR = {avg_meteor:.3f}, BERTScore = {avg_bert:.3f}, STSscore = {avg_sts:.3f},"
                  f"NDCG_METEOR = {avg_ndcg_meteor:.3f}, NDCG_BERTScore = {avg_ndcg_bert:.3f}")

    # Save results to a CSV file
    results_df = pd.DataFrame(results, columns=["RunFile", "METEOR", "BERTScore", "STSscore", "NDCG_METEOR", "NDCG_BERTScore", "DiagnosticFile", "PerResponseFile"])
    results_csv_path = os.path.join(run_folder, "evaluation_results.csv")
    results_df.to_csv(results_csv_path, index=False, float_format="%.3f")

    print(f"\nFinal results saved to: {results_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple runs using METEOR, BERTScore, and NDCG")
    parser.add_argument("ground_truth", help="Path to the ground truth CSV file")
    parser.add_argument("run_folder", help="Path to the folder containing run files")

    args = parser.parse_args()

    process_folder(args.ground_truth, args.run_folder)
