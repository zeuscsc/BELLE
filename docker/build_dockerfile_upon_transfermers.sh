# export https_proxy=...
# export http_proxy=...
# export all_proxy=...

wget https://raw.githubusercontent.com/huggingface/transformers/main/docker/transformers-pytorch-deepspeed-latest-gpu/Dockerfile -O transformers.dockerfile
# docker build --network host --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --build-arg all_proxy=$all_proxy -t transformers:ds -f transformers.dockerfile .
# docker build --network host --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --build-arg all_proxy=$all_proxy -t belle -f belle.dockerfile .

docker build --network host -t transformers:ds -f transformers.dockerfile .
docker build --network host -t belle -f belle.dockerfile .