from _old.eda import EDA

def main():
    eda = EDA()
    #eda.amount_sample()
    #eda.monthly_transactions()
    #eda.patterns()
    #eda.errors()
    #eda.correlations()
    eda.anomalies()
    eda.summary()
    del eda
    
if __name__ == "__main__":
    main()