// build.sbt
name := "SparkMLDemo"
version := "1.0"
scalaVersion := "2.12.15"  // 需与Spark版本兼容（查看Spark官方文档）

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.3.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.3.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.3.0" % "provided"
)