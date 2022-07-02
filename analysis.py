import pandas as pd

def main():
    df = pd.read_csv("neo.csv")
    
    print(df.describe())

if __name__ == "__main__":
    main()