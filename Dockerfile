FROM python:3.11.10

# copy model weights to avoid unnecessary download 

WORKDIR /JointPoseEstimation

COPY src/requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./src ./src 
COPY ./models ./models
COPY ./datasetss ./datasets
COPY ./utils ./utils
COPY ./src/10-18-2024_70.pth ./src

EXPOSE 3000

CMD ["python", "src/app.py"]



