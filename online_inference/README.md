~~~
docker build -t vkorennoy/online_inference:v1 .
docker run -p 8000:8000 vkorennoy/online_inference:v1
python make_request.py 

~~~