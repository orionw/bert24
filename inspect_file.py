import pyarrow.parquet as pq
import pandas as pd

# Read the parquet file
file_path = '/brtx/archive/orionw/our_bert24/bert24/data_order/checkpoint_000001/steps_000001_to_000010.parquet'
table = pq.read_table(file_path)

# Convert to pandas for easier inspection
df = table.to_pandas()

# Print basic info
print("Schema:")
print(table.schema)
print("\nShape:", df.shape)
print("\nFirst few rows:")
print(df.head())
breakpoint()

# Print metadata (which should include sequence length info)
print("\nMetadata:")
for key, value in table.schema.metadata.items():
    print(f"{key.decode()}: {value.decode()}")