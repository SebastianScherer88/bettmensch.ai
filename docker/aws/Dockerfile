# as taken from:
# https://github.com/aws/aws-cli/issues/3291#issuecomment-453197195
FROM python:3.11

ARG CLI_VERSION=1.16.86

RUN pip install --upgrade pip --no-cache-dir boto3

WORKDIR /aws

CMD sh