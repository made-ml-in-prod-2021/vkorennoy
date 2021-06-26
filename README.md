# Course: ML in production
## Homework-4

#### Author: Viktor Korennoy (DS-21)

Project is based on google cloud system:  
https://console.cloud.google.com/


------------
Create docker image
```
cd online_inference
docker build -t vkorennoy/online_inference:v1 .
docker run -p 8000:8000 vkorennoy/online_inference:v1
cd ./..
```

Check if cluster works
```
kubectl cluster-info
```

Run pods:
```
kubectl apply -f kubernetes/online-inference-pod.yaml
kubectl apply -f kubernetes/online-inference-pod-probes.yaml
kubectl apply -f kubernetes/online-inference-pod-resources.yaml
kubectl apply -f kubernetes/online-inference-replicaset.yaml
kubectl apply -f kubernetes/online-inference-deployment-blue-green.yaml
kubectl apply -f kubernetes/online-inference-deployment-rolling-update.yaml
```

Enable port forwarding
```
kubectl port-forward pod/online-inference-pod 8000:8000
kubectl apply -f kubernetes/online-inference-pod-probes 8000:8000
...
```

Get logs
```
kubectl describe pod online-inference-pod
kubectl describe pod online-inference-pod-probes
...
```
