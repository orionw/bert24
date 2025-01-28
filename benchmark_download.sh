rm -rf c4-36b-baseline/
time python download.py --pattern "*" --repo orionweller/c4-36b-decompressed --output c4-36b-baseline/ --workers 40

# 120 workers: 3:00
# 8 workers: 8:30
# 40 worker: 2:30
