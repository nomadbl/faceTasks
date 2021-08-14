 #!/bin/bash
echo "Copy s3 privileges"
addr=$1
scp -i ~/.ssh/lior.pem -r ~/.aws ec2-user@$(addr):~/
scp -i ~/.ssh/lior.pem session.zip ec2-user@$(addr):~/session.zip
ssh -L localhost:8888:localhost:8888 -i ~/.ssh/lior.pem ec2-user@$(addr)
echo "Copy data from s3..."
aws s3 cp s3://nomadblfaces/data data.zip
echo "Copied data from s3"
echo "unpacking data"
unzip data.zip
echo "unpacking session"
unzip session.zip
echo "starting jupyter"
jupyter notebook &
disown -h %1
