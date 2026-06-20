import io
import os
import re
import sqlite3
import sys
from contextlib import closing

# Ensure project root is on PYTHONPATH
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from app import app, init_db, THRESHOLD


def pick_sample_cvs(n=2):
    paths = []
    for fn in os.listdir('uploads'):
        if fn.lower().endswith(('.pdf', '.docx')):
            paths.append(os.path.join('uploads', fn))
    return paths[:n]


def run_smoke():
    init_db()
    client = app.test_client()

    files = []
    for p in pick_sample_cvs(2):
        with open(p, 'rb') as f:
            files.append((io.BytesIO(f.read()), os.path.basename(p)))

    # Analyze
    data = {
        'job_role': 'Software Engineer',
        'files': files,
    }
    resp = client.post('/analyze', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200, f"Analyze failed: {resp.status_code}"
    html = resp.data.decode('utf-8', errors='ignore')
    print('[analyze] page length:', len(html))

    # Extract candidate names from table rows
    names = re.findall(r'<td class=\"fw-semibold\">([^<]+)</td>|<td>(CV_[^<]+)</td>', html)
    flat_names = [a or b for a, b in names if (a or b)]

    # DB check
    with closing(sqlite3.connect('candidates.db')) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), SUM(CASE WHEN final_score >= ? THEN 1 ELSE 0 END) FROM candidates", (THRESHOLD,))
        total, suitable = cur.fetchone()
        print(f"[db] total={total}, suitable={suitable}, threshold={THRESHOLD}")

        cur.execute("SELECT name, final_score FROM candidates ORDER BY final_score DESC LIMIT 1")
        row = cur.fetchone()
        assert row, "No candidates inserted"
        top_name = row[0]

    # Personality test page
    resp = client.get(f'/test/{top_name}')
    assert resp.status_code in (200, 302), f"Test page failed: {resp.status_code}"

    # Submit personality test with neutral-to-positive answers
    form = {f"q{i}": str(4 if i % 2 == 0 else 3) for i in range(1, 11)}
    resp = client.post(f'/submit_test/{top_name}', data=form, follow_redirects=True)
    assert resp.status_code == 200, f"Submit test failed: {resp.status_code}"

    # Ranking page
    resp = client.get('/ranking')
    assert resp.status_code == 200, f"Ranking failed: {resp.status_code}"
    print('[ranking] page length:', len(resp.data))

    print('SMOKE TEST: OK')


if __name__ == '__main__':
    run_smoke()
