package dm.sns.algo

import java.io.{BufferedOutputStream, File, FileOutputStream, PrintWriter}
import java.net.URI
import java.nio.file.{Files, Paths}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.language.postfixOps
import scala.sys.process._
import scala.util.{Random, Try}

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import dm.sns.utils.HdfsUtil

class LabelPropagationMetis(val numParts: Int) extends Serializable {

  @transient lazy val logger: Logger = Logger.getLogger(classOf[LabelPropagationMetis])
  private val refineStart: Int = 1
  private var round = 5
  private var lpIterations = 3
  private var nbrUpper = 100
  private var wRatio = 0.1
  private var saveRatio = 0.2
  private var numPartitions: Int = 100
  private var metisAppPath: String = null
  private var outputPath: String = null

  def config(round: Int, lpIterations: Int, nbrUpper: Int, wRatio: Double, saveRatio: Double): this.type = {
    this
      .setRound(round)
      .setLPIterations(lpIterations)
      .setNbrUpperSize(nbrUpper)
      .setWRatio(wRatio)
      .setSaveRatio(saveRatio)
  }

  def setRound(round: Int): this.type = {
    require(round > 0, s"round must be positive.")
    this.round = round
    this
  }

  def setLPIterations(lpIterations: Int): this.type = {
    require(lpIterations > 0, s"lpIterations must be positive.")
    this.lpIterations = lpIterations
    this
  }

  def setNbrUpperSize(nbrUpper: Int): this.type = {
    require(nbrUpper > 0, s"nbrUpperSize must be positive.")
    this.nbrUpper = nbrUpper
    this
  }

  def setWRatio(wRatio: Double): this.type = {
    require(0 < wRatio && wRatio < 1, s"wRatio must be between 0 and 1.")
    this.wRatio = wRatio
    this
  }

  def setSaveRatio(saveRatio: Double): this.type = {
    require(0 < saveRatio && saveRatio < 1, s"saveRatio must be between 0 and 1.")
    this.saveRatio = saveRatio
    this
  }

  def setDataPartition(dataPartition: Int): this.type = {
    this.numPartitions = dataPartition
    this
  }

  def setMetisAppPath(path: String): this.type = {
    this.metisAppPath = path
    this
  }

  def setOutputPath(path: String): this.type = {
    this.outputPath = path
    this
  }

  def run(edgeRDD: RDD[(Long, Long, Float)]): RDD[(String, String)] = {
    val coarenRDD = this.coaren(edgeRDD)
    val metisRDD = this.localMetis(coarenRDD)
    val partitionRDD = this.refine(metisRDD)
    partitionRDD
  }

  def coaren(edgeRDD: RDD[(Long, Long, Float)]): RDD[(Long, Int, Long, Float)] = {
    val sc = edgeRDD.sparkContext
    logger.info("edge count = " + edgeRDD.count())

    var edge = edgeRDD
      .map { case (u, v, w) => (u, (v, w)) }
      .groupByKey(numPartitions)
      .flatMap {
        case (u, iter) => {
          val arr = iter.toArray
          val sumNbrw = arr.map(_._2).sum
          val sr =
            if (this.nbrUpper / arr.length.toDouble < this.saveRatio)
              this.nbrUpper / arr.length.toDouble
            else this.saveRatio
          arr
            .filter(x => x._2 / sumNbrw > this.wRatio || math.random < sr)
            .map(x => (x._1, (u, x._2)))
        }
      }

    logger.info(s"========== coaren step =========")
    for (r <- 1 to this.round) {
      logger.info(s"  ========== round $r =========")
      var nodeId = edge
        .flatMap { case (v, (u, weight)) => Array(u, v) }
        .distinct(numPartitions)
        .map(id => (id, id))

      val partitionSizeUpper = nodeId.count() / this.numParts

      for (i <- 1 to this.lpIterations) {

        val sizeUpper = nodeId
          .map(x => (x._2, 1L))
          .reduceByKey(_ + _, numPartitions)
          .map { case (pid, pidn) =>
            (pid, (pidn >= partitionSizeUpper, pidn))
          }

        val nodeFreeze = nodeId
          .map(_.swap)
          .join(sizeUpper, numPartitions)
          .map { case (pid, (id, (fpid, pidn))) => (id, (pid, fpid, pidn)) }

        nodeId = edge
          .join(nodeFreeze, numPartitions)
          .map { case (v, ((u, w), (vpid, vfpid, vpidn))) =>
            (u, (v, (vpid, vfpid, vpidn), w))
          }
          .groupByKey(numPartitions)
          .map {
            case (u, iter) => {
              val vpidMap = scala.collection.mutable.Map[Long, Float]()
              iter.toArray
                .filter(!_._2._2)
                .map {
                  case (_, (vpid, _, vpidn), w) => {
                    if (!vpidMap.contains(vpid)) vpidMap(vpid) = w / vpidn
                    else vpidMap(vpid) = vpidMap(vpid) + w / vpidn
                  }
                }
              val vpidArr = vpidMap.toArray
                .sortWith((x, y) => {
                  if (x._2 == y._2) x._1 < y._1 else x._2 > y._2
                })
              val pid =
                if (vpidArr.length == 0) -1L else vpidArr(0)._1
              (u, pid)
            }
          }
          .join(nodeFreeze, numPartitions)
          .map {
            case (u, (newpid, (oldpid, fpid, _))) => {
              if (fpid || newpid == -1L) (u, oldpid)
              else (u, newpid)
            }
          }
      }

      val midPath = this.outputPath + s"/node_round${r}"
      if (HdfsUtil.exists(sc, midPath)) HdfsUtil.delete(sc, midPath)
      nodeId.map { case (u, pid) => u.toString + "," + pid.toString }.saveAsTextFile(midPath)

      val newEdge = edge
        .join(nodeId, numPartitions)
        .map { case (v, ((u, w), vpid)) => (u, (vpid, w)) }
        .join(nodeId, numPartitions)
        .map { case (u, ((vpid, w), upid)) =>
          ((upid, vpid), (w, 1f))
        }
        .filter(x => x._1._1 != x._1._2)
        .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2), numPartitions)

      edge = newEdge
        .map { case ((upid, vpid), (w, n)) => (upid, (vpid, w, n)) }
        .groupByKey(numPartitions)
        .flatMap {
          case (upid, iter) => {
            val arr = iter.toArray
            val sumNbrw = arr.map(_._2).sum
            val sr =
              if (this.nbrUpper / arr.length.toDouble < this.saveRatio)
                this.nbrUpper / arr.length.toDouble
              else this.saveRatio
            arr
              .filter(x => x._2 / sumNbrw > this.wRatio || math.random < sr)
              .map(x => (x._1, (upid, x._2)))
          }
        }
    }

    // id, pid
    var rdd = sc
      .textFile(this.outputPath + s"/node_round${this.round}")
      .map { line => (line.split(",")(0), line.split(",")(1)) }

    for (r <- (this.refineStart to this.round - 1).reverse) {
      val trans = sc
        .textFile(this.outputPath + s"/node_round${r}")
        .map { line => (line.split(",")(0), line.split(",")(1)) }
      rdd = trans
        .join(rdd, numPartitions)
        .map { case (mid, (id, pid)) => (id, pid) }
    }

    val nodeW = rdd.map(x => (x._2.toLong, 1)).reduceByKey(_ + _, numPartitions)

    edge
      .map { case (v, (u, w)) => ((u, v), w) }
      .flatMap(x => Array(x, (x._1.swap, x._2)))
      .reduceByKey((x, y) => if (x < y) y else x, numPartitions)
      .map(x => (x._1._1, (x._1._2, x._2)))
      .leftOuterJoin(nodeW, numPartitions)
      .map { case (u, ((v, w), uw)) => (u, uw.getOrElse(1), v, w) }
  }

  def localMetis(edgeRDD: RDD[(Long, Int, Long, Float)]): RDD[(Long, Int)] = {

    val sc = edgeRDD.sparkContext
    logger.info(s"========== run localMetis =========")

    val edges = edgeRDD
      .map(x => (x._1, (x._3, x._4)))
      .persist(StorageLevel.MEMORY_AND_DISK_SER)

    val nodes = edges
      .map { case (u, _) => u }
      .distinct(numPartitions)
      .zipWithIndex()
      .map { case (u, idx) => (u, idx.toInt + 1) }
      .persist(StorageLevel.MEMORY_AND_DISK_SER)
      .setName("nodes")

    val updatedEdges = edges
      .join(nodes, numPartitions)
      .map { case (_, ((v, w), uidx)) => (v, (uidx, w)) }
      .join(nodes, numPartitions)
      .map { case (_, ((uidx, w), vidx)) => (uidx, (vidx, w)) }
      .groupByKey(numPartitions)
      .mapValues(x => x.toArray.sortBy(x => x._1).map(x => f"${x._1}").mkString(" "))
      .collectAsMap()

    val updatedNodes = edgeRDD
      .map(x => (x._1, x._2))
      .distinct(numPartitions)
      .join(nodes, numPartitions)
      .map { case (_, (uw, idx)) => (idx, uw) }
      .collectAsMap()

    val HDFS = "hdfs://" + metisAppPath.split("/")(2)
    val fs =
      if (metisAppPath != null && metisAppPath.nonEmpty)
        FileSystem.get(URI.create(HDFS), sc.hadoopConfiguration, null)
      else FileSystem.get(URI.create(HDFS), sc.hadoopConfiguration)
    val path = new Path(metisAppPath)
    val metisApp =
      if (fs.exists(path)) {
        val conf = sc.hadoopConfiguration
        val fileSystem = path.getFileSystem(conf)
        val reader = fileSystem.open(path)
        val dataBytes = Array.fill(reader.available())(Byte.MinValue)
        reader.readFully(dataBytes)
        dataBytes
      } else null

    if (metisApp == null) return sc.emptyRDD[(Long, Int)]

    val inputfilename = s"metis_file_${System.currentTimeMillis()}_${Math.abs(Random.nextLong())}"
    val outputfilename = s"$inputfilename.part.$numParts"
    val appName = "gpmetis"

    if (Files.exists(Paths.get(appName))) Files.delete(Paths.get(appName))
    val file = new File(appName)
    val appWriter = new BufferedOutputStream(new FileOutputStream(file))
    appWriter.write(metisApp)
    appWriter.close()
    file.setExecutable(true, false)
    file.setWritable(true, false)
    file.setReadable(true, false)

    val numNodes = nodes.count().toInt
    val numEdges = edges.count().toInt
    logger.info(s"$numNodes $numEdges=${numEdges / 2}")
    val writer = new PrintWriter(new File(inputfilename))

    // 010: use vertex weight
    writer.write(s"$numNodes ${numEdges / 2} 010\n")
    (1 to numNodes).foreach { u =>
      val nbr = updatedEdges.getOrElse(u, "")
      val w = updatedNodes.getOrElse(u, 1)
      writer.write(s"$w $nbr\n")
    }
    writer.flush()
    writer.close()
    var retryCnt = 3
    while (retryCnt > 0) {
      retryCnt -= 1
      Try {
        val gpmetisPrints = (s"./$appName $inputfilename $numParts" !!)
        if (gpmetisPrints.nonEmpty) logger.info(s"[done] ./$appName $inputfilename $numParts")
        retryCnt = 0
      }
    }
    Try(s"rm $inputfilename" !!).getOrElse(logger.info(s"cannot rm $inputfilename!"))
    val localPartition = Try {
      val idMap: ArrayBuffer[Int] = ArrayBuffer.empty[Int]
      for (line <- Source.fromFile(outputfilename).getLines()) {
        if (line.nonEmpty && line.charAt(0) != '#') idMap.append(line.toInt)
      }
      nodes.map { case (u, i) => (u, idMap(i - 1)) }
    }.getOrElse(sc.emptyRDD[(Long, Int)])
    nodes.unpersist(blocking = false)
    edges.unpersist(blocking = false)
    Try((s"rm $appName" !!)).getOrElse(logger.info(s"cannot rm $appName!"))

    localPartition
  }

  def refine(nodeId: RDD[(Long, Int)]): RDD[(String, String)] = {
    val sc = nodeId.sparkContext
    var rdd = nodeId.map(x => (x._1.toString, x._2.toString))
    logger.info(s"========== refine step =========")
    for (r <- (this.refineStart to this.round).reverse) {
      val trans = sc
        .textFile(this.outputPath + s"/node_round${r}")
        .map { line => (line.split(",")(0), line.split(",")(1)) }
      if (r > this.refineStart) {
        rdd = trans
          .join(rdd, numPartitions)
          .map { case (_, (id, pid)) => (id, pid) }
      } else {
        rdd = trans
          .leftOuterJoin(rdd, numPartitions)
          .map { case (_, (id, pidOps)) =>
            (id, pidOps.getOrElse(Random.nextInt(numParts).toString))
          }
      }
    }
    rdd
  }
}
