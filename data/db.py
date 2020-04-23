import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///data/DisasterResponse.db')

table = engine.table_names()
print(table)
con = engine.connect()

# Perform query: rs
rs = con.execute("SELECT * FROM Disaster_ETL")

# Save results of the query to DataFrame: df
df = pd.DataFrame(rs.fetchall())
df.columns = rs.keys()

# Close connection
con.close()

# Print head of DataFrame df
print(df.head())