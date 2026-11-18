import sys, time
sys.path.insert(0,'backend')
from app import app

print('=== LATENCY SMOKE TEST ===')
with app.test_client() as client:
    text = 'India signed 250 documents Space cooperation with 59 countries ISRO chief'
    t0 = time.perf_counter()
    r = client.post('/api/analyze', json={'text': text})
    ms1 = (time.perf_counter() - t0) * 1000
    print('First call (live fetch): ' + str(int(ms1)) + ' ms  status=' + str(r.status_code))
    t0 = time.perf_counter()
    r2 = client.post('/api/analyze', json={'text': text})
    ms2 = (time.perf_counter() - t0) * 1000
    print('Second call (cached):    ' + str(int(ms2)) + ' ms  status=' + str(r2.status_code))
    assert r.status_code == 200
    assert r2.status_code == 200

print('')
print('=== CORRECTNESS ===')
test_cases = [
    ('FAKE', 'SHOCKING: Scientists confirm hot water cures all cancers, doctors hate this!'),
    ('FAKE', 'Fact Check: No, PM Modi has not announced another Covid-19 lockdown'),
    ('FAKE', 'Viral video shows Modi performing black magic, shared widely on WhatsApp'),
    ('REAL', 'Central bank keeps interest rates unchanged, citing stable inflation'),
    ('REAL', 'LAC tensions: China strategy behind deliberately failing talks with India'),
    ('REAL', 'Hathras case CBI reaches victim village visits crime scene'),
]
correct = 0
with app.test_client() as client:
    for actual, text in test_cases:
        r = client.post('/api/analyze', json={'text': text})
        d = r.get_json()
        v = d['verdict']
        pred = 'REAL' if v == 'Likely real' else ('FAKE' if v == 'Likely fake' else 'UNV')
        ok = pred == actual
        if ok: correct += 1
        se = len(d.get('source_evidence', []))
        mark = 'OK   ' if ok else 'WRONG'
        print('[' + mark + '] actual=' + actual + '  pred=' + pred + '  cred=' + str(d['credibility']) + '/100  evidence=' + str(se) + '  ' + text[:55])

print('')
print('Score: ' + str(correct) + '/' + str(len(test_cases)))

print('')
print('=== NEW FILES ===')
from pathlib import Path
for f in ['backend/retriever.py','backend/similarity.py','backend/evaluate_grounded.py','models/grounded_metrics.json']:
    p = Path(f)
    print(('OK  ' if p.exists() else 'MISS') + ' ' + f + '  (' + str(p.stat().st_size if p.exists() else 0) + ' bytes)')
