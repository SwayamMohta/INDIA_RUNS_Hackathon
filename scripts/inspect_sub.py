import csv
with open('test_submission.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print('Total rows:', len(rows))
print()
print('Top 10 candidates:')
for r in rows[:10]:
    print(f"  #{r['rank']:>3} {r['candidate_id']}  score={r['score']}  {r['reasoning'][:90]}")
