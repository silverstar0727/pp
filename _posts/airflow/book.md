# ch1
날씨에 따른 판매 예측 (ML 데이터 파이프라인)

1. 판매 데이터
    1.1. 판매 데이터 가져오기
    1.2. 판매 데이터 변환
2. 날씨 데이터
    2.1. 날씨 데이터 가져오기
    2.2. 날씨 데이터 변환
3. 테스트 데이터셋 생성
4. 모델 훈련
5. 모델 배포

# ch2
* PythonOperator
* BashOperator
* EmailOperator
* OracleOperator
* BaseOperator
* SimpleHTTPOperator

## on prem
~~~sh
airflow db init
airflow users create \
--username admin \
--password admin \
--firstname Anonymous \
--lastname Admin \
--role Admin \
--email silverstar456@yonsei.ac.kr
cp dag2.py ~/airflow/dags/
airflow webserver
airflow scheduler
~~~

## in docker
~~~sh
docker run \
-it \
-p 8080:8080 \
-v dag2.py:/opt/airflow/dags/dag2.py \
--entrypoint=/bin/bash \
--name airflow \
apache/airflow:2.0.0-python3.8 \
-c '( \
airflow db init && \
airflow users create \
--username admin \
--password admin \
--firstname Anonymous \
--lastname Admin \
--role Admin \
--email silverstar456@yonsei.ac.kr \
); \
airflow webserver & \
airflow scheduler'
~~~

# ch3
~~~sh
curl -o /tmp/events.json http://localhost:5000/events
~~~