import requests

resp = requests.post("https://getprediction-bun42n2aya-rj.a.run.app", files={"file": open('./model/three.png','rb')})
#https://getprediction-bun42n2aya-rj.a.run.app

print(resp.json())