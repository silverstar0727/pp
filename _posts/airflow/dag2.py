import json
import pathlib

import airflow
import requests
import requests.exceptions as requests_exceptions
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

dag = DAG(
    dag_id="dowload_rocket_launches", # Dag 이름(필수)
    start_date=airflow.utils.dates.days_ago(14), # 실행 날짜(필수)
    schedule_interval=None, # 실행 간격(None: 자동으로 실행되지 않도록 설정)
)

# 다운로드를 받는 bash operator
download_launches = BashOperator(
    task_id="download_launches",
    bash_command="curl -o /tmp/launches.json -L 'https://11.thespacedevs.com/2.0.0/launch/upcoming'",
    dag=dag,
)

# 이미지를 크롤링하는 함수
def _get_pictures():
    # 출력 디렉토리가 있는지 확인하고 없으면 생성
    pathlib.Path("/tmp/images").mkdir(parents=True, exist_ok=True)


    with open("/tmp/launches.json") as f: 
        launches = json.load(f)
        image_urls = [launch['images'] for launch in launches['results']] # 이미지 url 추출
        
        # 모든 이미지 url에 대해서 실행
        for image_url in image_urls:
            try:
                response = requests.get(image_url)
                image_filename = image_url.split("/")[-1]
                target_file = f"/tmp/images/{image_filename}" # /tmp/images/{image_filename}에 저장
                with open(target_file, 'wb') as f:
                    f.write(response.content)
                # 저장된 파일과 위치를 출력
                print(f"Downloaded {image_url} to {target_file}")

            except requests_exceptions.MissingSchema:
                print(f"{image_url} appears to be an invalid URL")
            except requests_exceptions.ConnectionError:
                print(f"Couldn't connect to {image_url}")

# 함수 -> PythonOperator
get_pictures = PythonOperator(
    task_id="get_pictures",
    python_callable=_get_pictures,
    dag=dag
)

# 
notify = BashOperator(
    task_id="notify",
    bash_command='echo "There are now $(ls /tmp/images/ | wc -1) images."',
    dag=dag
)

download_launches >> get_pictures >> notify