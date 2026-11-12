VQA 2025 answer generation evaluation code:
python3 eval.vqa.ag.py <ground_truth> <runs_folder>

VQA 2025 multiple choice evaluation code:
python3 eval.vqa.mc.py <ground_truth> <runs_folder>

<ground_truth> file included and have the following format:
<query_ID>, <YouTube_ID>, <Question>, <correct_answer>

Example:
1,bI7D0GM1Ttc,"what happens when the clown gets on his knees?","the 3 dancers behind start to dance"

Please refer to the 2025 VQA track guidelines for run format in MC task:
https://www-nlpir.nist.gov/projects/tv2025/vqa.html

The scripts will generate:
- overall evaluation scores file (evaluation_results.csv)
- individual diagnostic file (showing results per query) for each run and submitted result item (saved in the <runs_folder> location).
- individual "per_response" file (showing results per submitted response for a query) for each run (saved in the <runs_folder> location).
