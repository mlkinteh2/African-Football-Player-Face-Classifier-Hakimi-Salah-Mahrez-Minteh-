import requests, json
try:
    r = requests.get('http://127.0.0.1:5000/health', timeout=5)
    print(r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)
except Exception as e:
    print('ERROR', e)
