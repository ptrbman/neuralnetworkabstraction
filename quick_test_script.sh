#! /bin/sh

docker build -t abstraction-test-environment .
docker run -it -p 8888:8888 -v $(pwd):/workspace abstraction-test-environment
