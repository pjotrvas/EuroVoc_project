import matplotlib.pyplot as plt
import pandas as pd


def main():
    log_to_load = input("Enter group: ").strip().lower()
    df_training = pd.read_csv(f'./csv_logs/log_{log_to_load}_train.csv')
    df_eval = pd.read_csv(f'./csv_logs/log_{log_to_load}_eval.csv')

    plt.plot(df_training.epoch, df_training.loss)
    plt.plot(df_eval.epoch, df_eval.eval_loss)
    plt.show()


main()
