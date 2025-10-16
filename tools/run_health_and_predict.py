import requests, json, sys

HEALTH = 'http://127.0.0.1:5000/health'
PRED = 'http://127.0.0.1:5000/predict'

try:
    r = requests.get(HEALTH, timeout=5)
    print('HEALTH_STATUS', r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print('HEALTH_TEXT', r.text)
except Exception as e:
    print('HEALTH_ERROR', e)
    sys.exit(1)

j = r.json()
if not j.get('model_loaded'):
    print('Model not loaded; aborting predict test.')
    sys.exit(0)

# If model loaded, try a sample image (Mohamed Salah path)
img_path = 'Model/Dataset/Mohamed Salah/Image_1.jpg'
try:
    with open(img_path, 'rb') as f:
        files = {'image': f}
        r2 = requests.post(PRED, files=files, timeout=10)
        print('PRED_STATUS', r2.status_code)
        try:
            print(json.dumps(r2.json(), indent=2))
        except Exception:
            print('PRED_TEXT', r2.text)
except FileNotFoundError:
    print('Sample image not found:', img_path)
except Exception as e:
    print('PRED_ERROR', e)
