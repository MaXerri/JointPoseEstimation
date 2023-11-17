import json
import boto3
from constants import BUCKET_NAME

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    prefix = event['UUID'] + '/'
    objects = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)['Contents']
    key = objects[0]['Key']
    file_name = key.split('/', 1)[1]
    file_loc = '/tmp/' + file_name
    s3.download_file(BUCKET_NAME, key, file_loc)
    upload_loc = event['UUID'] + '-jpe' '/'
    key = upload_loc + file_name
    s3.upload_file(file_loc, BUCKET_NAME,  key)
    
    url = s3.generate_presigned_url('get_object', Params={'Bucket': BUCKET_NAME, 'Key': key}, ExpiresIn=1800)
    response =  {
        'statusCode': 200,
        'headers' : {
            'Access-Control-Allow-Origin' : '*'
        },
        'body': url
    }
    return response
