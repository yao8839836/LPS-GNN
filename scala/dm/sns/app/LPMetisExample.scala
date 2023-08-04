package dm.sns.app

import org.apache.spark.sql.SparkSession

import dm.sns.algo.LabelPropagationMetis
import dm.sns.utils.{HdfsUtil, Logging, OptionParser}

object LPMetisExample {

  def main(args: Array[String]): Unit = {
    val options = new OptionParser(args)
    val sparkDriverMaxResultSize = options.getInt("spark_driver_maxResultSize", 40)

    val spark = SparkSession
      .builder()
      .appName("LabelPropagationMetis")
      .config("spark.driver.maxResultSize", f"${sparkDriverMaxResultSize}G")
      .getOrCreate()

    val edgeInputPath = options.getString("edgeInputPath")
    val edgeDelimiter = options.getString("edgeDelimiter", "\\s+")
    val partitionOutputPath = options.getString("partitionOutputPath").stripSuffix("/")

    val uIdx = options.getInt("uIdx")
    val vIdx = options.getInt("vIdx")
    val wIdx = options.getInt("wIdx", -1)

    val dataPartitions = options.getInt("dataPartitions")
    val numParts = options.getInt("numPart")

    val lpIterations = options.getInt("lpIterations", 3)
    val round = options.getInt("round", 3)
    val nbrUpper = options.getInt("nbrUpper", 100)
    val wRatio = options.getDouble("wRatio", 0.1)
    val saveRatio = options.getDouble("saveRatio", 0.2)

    val metisAppPath = options.getString("metisAppPath")

    // local-metis require undirect graph
    val edge = spark.sparkContext
      .textFile(edgeInputPath)
      .map { line =>
        {
          val arr = line.split(edgeDelimiter)
          val u = arr(uIdx).toLong
          val v = arr(vIdx).toLong
          val w = if (wIdx == -1) 1f else arr(wIdx).toFloat
          (u, v, w)
        }
      }

    Logging.loggingConfig(classOf[LabelPropagationMetis])

    val algo = new LabelPropagationMetis(numParts)
      .setDataPartition(dataPartitions)
      .setMetisAppPath(metisAppPath)
      .setOutputPath(partitionOutputPath)
      .config(round, lpIterations, nbrUpper, wRatio, saveRatio)

    val partitionRDD = algo.run(edge).map(pair => f"${pair._1}\t${pair._2}")

    val outPath = partitionOutputPath + "/partition"
    if (HdfsUtil.exists(spark.sparkContext, outPath)) HdfsUtil.delete(spark.sparkContext, outPath)
    partitionRDD.saveAsTextFile(outPath)
  }
}
