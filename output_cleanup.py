import pandas as pd
import sys

def main():
    out_file_path = sys.argv[1]

    results_training = {}
    results_eval = {}
    with open(out_file_path, 'r', encoding='UTF8') as f:
        for line in f:
            if line.startswith('{'):
                tmp = eval(line)

                if not len(results_eval) and 'eval_loss' in tmp:
                    results_eval.update({k: [v] for k, v in tmp.items()})

                elif 'eval_loss' in tmp:
                    for k, v in tmp.items():
                        results_eval[k].append(v)

                elif not len(results_training) and 'loss' in tmp:
                    results_training.update({k: [v] for k, v in tmp.items()})

                elif 'loss' in tmp:
                    for k, v in tmp.items():
                        results_training[k].append(v)

        df_training = pd.DataFrame.from_dict(results_training)
        df_eval = pd.DataFrame.from_dict(results_eval)
        df_training.to_csv(f'./csv_logs/{out_file_path.split("/")[1]}_train.csv')
        df_eval.to_csv(f'./csv_logs/{out_file_path.split("/")[1]}_eval.csv')


main()
