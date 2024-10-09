import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items  # In case pandas>=2.0 and Spark<3.4
