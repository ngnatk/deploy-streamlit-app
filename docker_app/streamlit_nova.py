#!/usr/bin/env python
# -*- coding: utf8 -*-

# Example usage:
# - streamlit run streamlit_nova_canvas.py
# - streamlit run streamlit_nova_canvas.py --server.runOnSave True --server.port 8501

import amazon_video_util
import base64
import boto3
import config_file
import datetime
import file_utils
import json
import logging
import os
import random
import streamlit as st

from amazon_image_gen import BedrockImageGenerator
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError
from utils.auth import Auth

# Setup Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

## App configuration
st.set_page_config(
    page_title="Image and Video Generation with Amazon Nova",
    layout="wide"
)
st.title("🌇 Image and Video Generation with Amazon Nova 🎨")
output_folder = "output"

# Setup AWS
bedrock_runtime = boto3.client('bedrock-runtime')
s3_client = boto3.client("s3")
ssm = boto3.client('ssm')
dynamodb = boto3.client('dynamodb')
sts = boto3.client('sts')

AWS_ACCOUNT_ID = sts.get_caller_identity().get('Account')
response = ssm.get_parameter(Name='/streamlit/STREAMLIT_S3_BUCKET')
STREAMLIT_S3_BUCKET = response['Parameter']['Value']
response = ssm.get_parameter(Name='/streamlit/DYNAMODB_TABLE')
DYNAMODB_TABLE = response['Parameter']['Value']

# ID of Secrets Manager containing cognito parameters
secrets_manager_id = config_file.Config.SECRETS_MANAGER_ID

# ID of the AWS region in which Secrets Manager is deployed
region = config_file.Config.DEPLOYMENT_REGION

# Initialise CognitoAuthenticator
authenticator = Auth.get_authenticator(secrets_manager_id, region)

# Authenticate user, and stop here if not logged in
is_logged_in = authenticator.login()
if not is_logged_in:
    logging.info("st.stop()")
    st.stop()


def logout():
    logging.info("authenticator.logout()")
    authenticator.logout()


with st.sidebar:
    st.text(f"Welcome,\n{authenticator.get_username()}")
    st.button("Logout", "logout_btn", on_click=logout)



def generate_image(prompt, negative_prompt, quality="standard", seed=None, width=1280, height=720, cfgScale=7.0, num_images=1):
    """
    Generate images using Nova Canvas
    """

    if seed is None:
        seed = random.randint(0, 858993459)

    inference_params = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
            "negativeText": negative_prompt,
        },
        "imageGenerationConfig": {
            "numberOfImages": num_images,  # Number of variations to generate. 1 to 5.
            "quality": quality,  # Allowed values are "standard" and "premium"
            "width": width,  # See README for supported output resolutions
            "height": height,  # See README for supported output resolutions
            "cfgScale": cfgScale,  # How closely the prompt will be followed
            "seed": seed
        },
    }

    # Define an output directory with a unique name.
    generation_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = f"output/{generation_id}"

    # Create the generator.
    generator = BedrockImageGenerator(output_directory=output_directory)
    response = generator.generate_images(inference_params)

    if "images" in response:
        image = file_utils.save_base64_images(response["images"], output_directory, "image")
    return response, image



def remove_background(source_image_base64, source_image_path=None):
    """
    Removes background from an image. Adapted from:
    https://github.com/aws-samples/amazon-nova-samples/blob/main/multimodal-generation/image-generation/python/07_background_removal.py
    """

    if source_image_path is not None:
        with open(source_image_path, "rb") as image_file:
            source_image_base64 = base64.b64encode(image_file.read()).decode("utf8")

    inference_params = {
        "taskType": "BACKGROUND_REMOVAL",
        "backgroundRemovalParams": {
            "image": source_image_base64,
        },
    }

    generation_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = f"output/{generation_id}"
    generator = BedrockImageGenerator(output_directory=output_directory)
    response = generator.generate_images(inference_params)

    if "images" in response:
        image = file_utils.save_base64_images(response["images"], output_directory, "image")
    return response, image


# The following subroutines: generate_video(), check_job_status(), list_job_statuses(),
# and monitor_recent_jobs() were adapted from:
# https://github.com/aws-samples/amazon-nova-samples/blob/main/multimodal-generation/video-generation/python/01_text_to_video_generation.py

def generate_video(s3_destination_bucket, video_prompt, model_id="amazon.nova-reel-v1:0"):
    """
    Generate a video using the provided prompt.
    
    Args:
        s3_destination_bucket (str): The S3 bucket where the video will be stored
        video_prompt (str): Text prompt describing the desired video
    """

    model_input = {
        "taskType": "TEXT_VIDEO",
        "textToVideoParams": {
            "text": video_prompt,
        },
        "videoGenerationConfig": {
            "durationSeconds": 6,  # 6 is the only supported value currently
            "fps": 24,  # 24 is the only supported value currently
            "dimension": "1280x720",  # "1280x720" is the only supported value currently
            "seed": random.randint(
                0, 2147483648
            ),  # A random seed guarantees we'll get a different result each time this code runs
        },
    }

    try:
        # Start the asynchronous video generation job
        invocation = bedrock_runtime.start_async_invoke(
            modelId=model_id,
            modelInput=model_input,
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{s3_destination_bucket}"}},
        )

        # Store the invocation ARN
        invocation_arn = invocation["invocationArn"]

        # Pretty print the response JSON
        logger.info("\nResponse:")
        logger.info(json.dumps(invocation, indent=2, default=str))

        # Save the invocation details for later reference
        amazon_video_util.save_invocation_info(invocation, model_input)

        return invocation_arn

    except Exception as e:
        logger.error(e)
        st.error(f"Error: {e}")
        return None



def check_job_status(invocation_arn):
    """Check status of a specific job using get_async_invoke()"""
    try:
        response = bedrock_runtime.get_async_invoke(
            invocationArn=invocation_arn
        )
        
        status = response["status"]
        logger.info(f"Status: {status}")
        logger.info("\nFull response:")
        logger.info(json.dumps(response, indent=2, default=str))
        return response
    except Exception as e:
        logger.error(f"Error checking job status: {e}")
        return None



def list_job_statuses(max_results=10, status_filter="InProgress"):
    """List all video generation jobs with optional filtering"""
    try:
        invocation = bedrock_runtime.list_async_invokes(
            maxResults=max_results,
            statusEquals=status_filter,
        )
        
        logger.info("Invocation Jobs:")
        logger.info(json.dumps(invocation, indent=2, default=str))
        return invocation
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return None



def monitor_recent_jobs(duration_hours=1):
    """Monitor and download videos from the past N hours"""
    from_submit_time = datetime.datetime.now(datetime.timezone.utc) \
        - datetime.timedelta(hours=duration_hours)
    return amazon_video_util.monitor_and_download_videos("output", submit_time_after=from_submit_time)



def job_json_from_arn(invocation_arn):
    job = bedrock_runtime.get_async_invoke(invocationArn=invocation_arn)
    arn = job['invocationArn']
    job_id = arn.split("/")[-1]
    status = job['status']
    submit_time = job['submitTime']
    try:
        s3_uri = job['outputDataConfig']['s3OutputDataConfig']['s3Uri']
    except Exception as e:
        s3_uri = ''

    return {
        'id': job_id,
        'submit_time': submit_time,
        'status': status,
        's3_uri': s3_uri
    }


def item_to_dynamodb(table, item):
    logging.info(f"item_to_dynamodb(): item={item}")
    if isinstance(item['submit_time'], datetime.date):
        item['submit_time'] = item['submit_time'].isoformat()

    try:
        response = dynamodb.put_item(
            TableName=table,
            Item={
                'id': {'S': item['id']},
                'prompt': {'S': item.get('prompt', '')},
                'submit_time': {'S': item.get('submit_time', '')},
                'status': {'S': item.get('status', '')},
                's3_uri': {'S': item.get('s3_uri' , '')}
            }
        )
    except ClientError as err:
        logger.error(
            f"DynamoDB: couldn't add item: {item}",
            table,
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise
    return response


def get_table_items(table_name, status_filter='InProgress'):
    table = boto3.resource('dynamodb').Table(table_name)
    response = table.scan(
        FilterExpression=Attr('status').eq(status_filter)
    )
    items = response['Items']
    return items




def tab_canvas():
    prompt = st.text_area("Enter your image prompt:", height=150)

    # Model parameters
    with st.sidebar:
        st.header("Model Parameters")
        negative_prompt = st.text_area("Negative Prompt:", 
            value="blurry, bad anatomy, bad hands, cropped, worst quality")
        quality = st.radio("Quality", options=["standard", "premium"])
        seed = st.number_input("Seed", min_value=0, value=42)
        width = st.sidebar.select_slider("Image Width", 
            options=[512, 768, 1024], value=512)
        height = st.select_slider("Image Height", 
            options=[512, 768, 1024], value=512)
        cfgRange = [1.1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
        cfgScale = st.select_slider("cfgScale", 
            options=cfgRange, value=7.0)

    if st.button("Generate Image"):
        if prompt:
            try:
                with st.spinner():
                    response, image = generate_image(prompt, negative_prompt, quality, seed, width, height, cfgScale)
                    
                    if 'images' in response:
                        # Decode and display the generated image
                        # image_data = base64.b64decode(response_body['images'][0])
                        # image = Image.open(io.BytesIO(image_data))
                        st.image(image, caption="Generated Image", use_container_width=True)
                    else:
                        st.error("No image was generated in the response")
                    
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
        else:
            st.warning("Please enter a prompt to generate an image")



def tab_background_removal():
    img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if img is not None:
        # Display the uploaded image
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Convert image to bytes, encode to base64
        img_bytes = img.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # Remove background and display the image
        with st.spinner():
            try:
                response, image = remove_background(img_base64)
            except Exception as e:
                st.error(f"Error removing background: {str(e)}")
                # st.info(json.dumps(response, indent=2, default=str))
            else:
                st.image(image, caption="Processed Image", use_container_width=True)



def tab_reel():
    # img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    video_prompt = st.text_area("Enter your video prompt:", height=150)

    if st.button("Generate Video"):
        if video_prompt:
            try:
                arn = generate_video(
                    s3_destination_bucket = STREAMLIT_S3_BUCKET,
                    video_prompt = video_prompt
                )
            except Exception as e:
                st.error(f"Error generating video: {str(e)}")
            else:
                st.session_state.invocation_arn = arn
                st.success(f"Video generation job started: {arn}")
                # st.markdown(f"```json\n{json.dumps(response, indent=2, default=str)}\n```")

            try:
                job_json = job_json_from_arn(arn)
                job_json['prompt'] = video_prompt
                logging.info(f"job_json = {job_json}")
                item_to_dynamodb(DYNAMODB_TABLE, job_json)

            except Exception as e:
                st.error(f"Error saving to dynamo db: {str(e)}")
        else:
            st.warning("Please enter a prompt to generate a video")



def tab_videos():
    # Check video status change from 'InProgress' to 'Completed'
    items_inprogress = get_table_items('NovaVideos', status_filter='InProgress')
    logging.info(f"len(items_inprogress) = {len(items_inprogress)}")
    for item in items_inprogress:
        id = item['id']
        status = item.get('status', '')
        prompt = item.get('prompt', '')
        submit_time = item.get('submit_time', '')
        if status == 'InProgress':
            if prompt:
                st.markdown(f'[InProgress] **Job ID**: `{id}`. **Submit time** = `{submit_time}`. **Prompt** = `{prompt}`')
            else:
                st.info(f'`[InProgress]` **Job ID**: `{id}`. **Submit time** = `{submit_time}`.')
            item_str = json.dumps(item, indent=2, default=str)
            logging.info(f"tab_videos(): item = {item_str}")
            arn = f'arn:aws:bedrock:us-east-1:{AWS_ACCOUNT_ID}:async-invoke/{id}'
            job_json = job_json_from_arn(arn)
            new_status = job_json['status']
            if new_status != status:
                item['status'] = new_status
                logging.info(f"{id}: {status} -> {new_status}")
                item_to_dynamodb(DYNAMODB_TABLE, item)
                # Copy the file fro the S3 bucket to temporary storage
                s3_client.download_file(STREAMLIT_S3_BUCKET, f'{id}/output.mp4', f'{id}.mp4')

    items_completed = get_table_items('NovaVideos', status_filter='Completed')
    logging.info(f"len(items_completed) = {len(items_completed)}")

    for i, item in enumerate(items_completed):
        invocation_id = item['id']
        status = item['status']
        prompt = item.get('prompt', '')
        filename = f"{invocation_id}.mp4"
        if prompt:
            if os.path.isfile(filename):
                with open(filename, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.info(f'"{prompt}" ({filename})')
                    st.video(video_bytes)
            else:
                logging.info(f"Video {i}: {filename} (not found)")
        else:
            logging.info(f"Video {i}: {filename} (prompt not found)")

        if i == 4:
            break



def main():
    if "invocation_arn" not in st.session_state:
        st.session_state.invocation_arn = ""

    tab1, tab2, tab3, tab4 = st.tabs(["Canvas", "Background Removal", "Reel", "Videos"])

    with tab1:
        tab_canvas()

    with tab2:
        tab_background_removal()

    with tab3:
        tab_reel()

    with tab4:
        tab_videos()



if __name__ == "__main__":
    main()
