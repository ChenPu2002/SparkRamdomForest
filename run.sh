docker run -d \
  --name spark-master \
  -h spark-master \
  -p 7077:7077 \
  -p 8080:8080 \
  --network spark-network \
  -v /home/ubuntu/data:/data \
  -e SPARK_MODE=master \
  bitnami/spark:latest

docker run -d \
  --name spark-worker \
  -h spark-worker \
  -p 8081:8081 \
  --network spark-network \
  -v /home/ubuntu/data:/data \
  -e SPARK_MODE=worker \
  -e SPARK_MASTER_URL=spark://spark-master:7077 \
  bitnami/spark:latest
  
cd ~/spark_project && \

docker run -it --rm \
  -v $(pwd):/app \
  -v /home/peter/.ivy2:/root/.ivy2 \
  -w /app \
  hseeberger/scala-sbt:8u312_1.6.2_2.12.15 \
  sbt clean assembly && \
  
cp target/scala-2.12/SparkMLDemo-assembly-1.0.jar ../spark/app.jar && \

docker exec spark-master \
  spark-submit \
  --master spark://spark-master:7077 \
  --class com.example.RandomForestDemo \
  /data/app.jar
