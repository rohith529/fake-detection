import sys; sys.path.insert(0,'backend')
from app import app

REQUIRED_OLD = {'verdict','credibility','p_real','model','latency_ms','markers','explanation','source','related','factchecks'}
REQUIRED_NEW = {'source_evidence','confidence_flag','source_signal'}

headlines = [
    ('FAKE', 'SHOCKING: Scientists confirm hot water cures all cancers, doctors hate this!'),
    ('FAKE', 'Fact Check: No, PM Modi has not announced another Covid-19 lockdown'),
    ('REAL', 'Central bank keeps interest rates unchanged, citing stable inflation and steady growth'),
    ('REAL', 'India signed 250 documents Space cooperation with 59 countries ISRO chief'),
]

correct = 0
with app.test_client() as client:
    r0 = client.post('/api/analyze', json={'text': headlines[0][1]})
    d0 = r0.get_json()
    missing_old = REQUIRED_OLD - d0.keys()
    missing_new = REQUIRED_NEW - d0.keys()
    assert not missing_old, 'Old fields missing: ' + str(missing_old)
    assert not missing_new, 'New fields missing: ' + str(missing_new)
    assert isinstance(d0['source_evidence'], list)
    assert isinstance(d0['confidence_flag'], bool)
    assert isinstance(d0['source_signal'], float)
    print('Field schema: OK')
    for actual, text in headlines:
        r = client.post('/api/analyze', json={'text': text})
        d = r.get_json()
        v = d['verdict']
        pred = 'REAL' if v == 'Likely real' else ('FAKE' if v == 'Likely fake' else 'UNV')
        ok = pred == actual
        if ok: correct += 1
        cred = d['credibility']
        cflag = d['confidence_flag']
        ssig = d['source_signal']
        mark = 'OK   ' if ok else 'WRONG'
        print('[' + mark + '] actual=' + actual + ' pred=' + pred + ' cred=' + str(cred) + '/100 flag=' + str(cflag) + ' src=' + str(ssig))
        for item in d['source_evidence']:
            assert all(k in item for k in ('title','link','source_domain','reliability','agreement'))
            assert 0.0 <= item['reliability'] <= 1.0
            assert 0.0 <= item['agreement'] <= 1.0
        assert 0 <= d['credibility'] <= 100
print('')
print('Score: ' + str(correct) + '/' + str(len(headlines)))
print('All end-to-end checks passed!')
