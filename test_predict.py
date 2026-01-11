import requests
url = 'http://127.0.0.1:8501/predict'
files = {'image': open(r'Model/test_imgaes/minteh.jpg','rb')}
try:
    r = requests.post(url, files=files, timeout=30)
    print('STATUS', r.status_code)
    print('TEXT', r.text[:1000])
except Exception as e:
    print('ERROR', e)
