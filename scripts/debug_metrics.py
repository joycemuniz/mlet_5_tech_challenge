from fastapi.testclient import TestClient
from src.api.app import app
client=TestClient(app)
print('GET /metrics before:')
print(client.get('/metrics').text)
print('GET /health:')
print(client.get('/health').status_code, client.get('/health').json())
class Dummy:
    def predict(self, X):
        return [0]
import src.api.app as _app_module
_app_module._model = Dummy()
print('POST /score status:', client.post('/score', json={'feature1':1}).status_code)
print('GET /metrics after:')
print(client.get('/metrics').text)
