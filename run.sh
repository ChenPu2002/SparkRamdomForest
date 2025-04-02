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