# Steps to Build and Run the ADVLM_VQA using Docker
## Build Docker
```
docker build -t advlmvqa_docker .
```
## RUN Docker
```
docker run --gpus all -it advlmvqa_docker /bin/bash
```

## Inside the Docker

### To run a demo
bash```
python -W ignore demo_vqa.py --image_path samples/sample01.png --prompts "what is the weather ? one word answer" "what is the lighting condition ? one word answer"
```
