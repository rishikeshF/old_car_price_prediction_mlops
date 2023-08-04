from time import sleep
from prefect_aws import S3Bucket, AwsCredentials 

# Create and save AWS credentials block
def create_aws_creds_block():
    # Create AWS credentials object 
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id= <insert_access_key>,
        aws_secret_access_key= <insert_secret_access_key>
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True) 

# Create and save S3 bucket block
def create_s3_bucket_block():
    # Load AWS credentials block
    aws_creds = AwsCredentials.load("my-aws-creds")

    # Create S3 bucket object 
    my_s3_bucket_obj = S3Bucket(
        bucket_name = "old-car-data-bucket",
        credentials= aws_creds
    )

    # Save the S3 bucket block
    my_s3_bucket_obj.save(name="s3-bucket-example", overwrite=True) 

if __name__ == "__main__": 
    # Create and save AWS credentials block
    create_aws_creds_block()

    # Sleep for 5 seconds to allow time for the AWS credentials to be saved
    sleep(5)

    # Create and save S3 bucket block
    create_s3_bucket_block() 

