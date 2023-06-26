import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(DATA_DIR, csv_name, PROB, eval):
    f = '{}/{}.csv'.format(DATA_DIR, csv_name)
    if PROB:
        f = '{}/{}.csv'.format(DATA_DIR, csv_name)

    df_ = pd.read_csv(f, header=None, names=['prob'])
    
    df_['id'] = eval['id'] 

    df_2 = pd.DataFrame()

    df_2['id'] = df_['id']
    df_2['prob'] = df_['prob']

    return df_2 

def main():
    train = pd.read_csv("/home/kaino/comp/month3/train.csv")

    eval = pd.read_csv("/home/kaino/comp/month3/sample_submit.csv")

    border = len(train[train["judgement"] == 1]) / len(train["judgement"])
    print(border)

    DATA_DIR = '/home/kaino/comp/month3'

    PROB = True

    TH = 0.020

    CSV_LIST = [
                    'predictions',
                    'predictions2',
                    'predictions3',
               ]

    WEIGHTS = [1, 0.5, 1]

    df = pd.DataFrame()

    i = 0
    for c in CSV_LIST:
        print(c)
        if len(df) == 0:
            df = read_data(DATA_DIR, c, PROB, eval)
            df.prob *= WEIGHTS[i]
        else:
            df_ = read_data(DATA_DIR, c, PROB, eval)
            # print(df_.prob.quantile(1-BORDER))
            df_.prob *= WEIGHTS[i]
            df = pd.concat([df, df_['prob']], axis=1)

        i += 1

    df.set_index('id', inplace=True)
    df.head()

    df['judgement'] = 0
    df['judgement'] = df.sum(axis=1)

    if PROB:
        # df['judgement'] = df.judgement / (len(CSV_LIST) * 1.0)
        df['judgement'] = df.judgement / (sum(WEIGHTS) * 1.0)
        df[['judgement']].to_csv('{}/sub.csv'.format(DATA_DIR), header=False)
        df['judgement'] = np.where(df.judgement < TH, 0, 1)
    else:
        # df['judgement'] = np.where(df.judgement < len(CSV_LIST) // 2, 0, 1)
        df['judgement'] = np.where(df.judgement < sum(WEIGHTS) // 2, 0, 1)

    df.reset_index(inplace=True) 
    print(df.judgement.sum())
    df.head(20)

    df[['id', 'judgement']].to_csv('{}/sub.csv'.format(DATA_DIR), index=False, header=False)

if __name__ == "__main__":
    main()