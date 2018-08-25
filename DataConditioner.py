import pandas

def col_standardize(df,include=[],exclude=[]):
    for col in [c for c in [list(df.columns) if len(include)==0 else include][0] if c not in exclude]:
        df[col]= (df[col]-df[col].mean())/df[col].std()