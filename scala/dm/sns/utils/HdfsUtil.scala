package dm.sns.utils

import java.io._
import java.net.URI
import java.util
import java.util.Comparator

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.util.Try

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileStatus, FileSystem, Path}
import org.apache.spark.SparkContext

object HdfsUtil {

  /**
   * 字符串写道hdfs指定路径上
   *
   * @param conf     hadoop配置变量，一般从sc.hadoopConfiguration获取
   * @param fileName hdfs文件路径
   * @param content  内容
   */
  def writeString(conf: Configuration, fileName: String, content: String): Unit = {
    val path = new Path(fileName)
    val fileSystem = path.getFileSystem(conf)

    val writer = new BufferedWriter(new OutputStreamWriter(fileSystem.create(path)))

    writer.write(content)
    writer.close()
  }

  /**
   * 读取指定hdfs文件中的内容
   *
   * @param conf     hadoop配置变量，一般从sc.hadoopConfiguration获取
   * @param fileName hdfs文件路径
   * @return sting数组，每行为一个元素
   */
  def readString(conf: Configuration, fileName: String): Array[String] = {
    val path = new Path(fileName)
    val fileSystem = path.getFileSystem(conf)

    val reader = new BufferedReader(new InputStreamReader(fileSystem.open(path)))

    val rst = new ArrayBuffer[String]

    var line = reader.readLine()
    while (line != null) {
      rst += line
      line = reader.readLine()
    }

    rst.toArray
  }

  /**
   * 将可序列化对象写入到指定hdfs路径
   *
   * @param conf     hdfs配置
   * @param fileName 路径
   * @param obj      需要写入对象
   */
  def writeObject(conf: Configuration, fileName: String, obj: AnyRef): Unit = {
    val path = new Path(fileName)
    val fileSystem = path.getFileSystem(conf)

    val writer = new ObjectOutputStream(fileSystem.create(path))
    writer.writeObject(obj)
    writer.close
  }

  /**
   * 读取hdfs上指定序列化对象
   *
   * @param conf     hdfs配置
   * @param fileName 路径
   * @return 对象引用
   */
  def readObject(conf: Configuration, fileName: String): AnyRef = {
    val path = new Path(fileName)
    val fileSystem = path.getFileSystem(conf)

    val reader = new ObjectInputStream(fileSystem.open(path))
    val result = reader.readObject
    reader.close

    result
  }

  /**
   * 获取文件的父目录
   *
   * @param path 文件路径
   * @return 父目录
   */
  def dirname(path: String): String = {
    new Path(path).getParent.toString
  }

  /**
   * 创建目录
   *
   * @param conf hadoop配置变量，一般从sc.hadoopConfiguration获取
   * @param path hdfs路径
   */
  def mkdir(conf: Configuration, path: String): Unit = {
    val p = new Path(path)
    val fileSystem = p.getFileSystem(conf)
    fileSystem.mkdirs(p)
  }

  /**
   * 删除目录
   *
   * @param conf hadoop配置变量，一般从sc.hadoopConfiguration获取
   * @param path hdfs路径
   */
  def rmdir(conf: Configuration, path: String): Unit = {
    val p = new Path(path)
    val fileSystem = p.getFileSystem(conf)
    fileSystem.delete(new Path(path), true)
  }

  /**
   * 根据文件大小，按时间由近到远取数据作为训练或测试数据
   *
   * @param conf          hadoop配置变量，一般从sc.hadoopConfiguration获取
   * @param trainDataPath 训练数据的路径
   * @param totalFileSize 总文件大小
   * @param validTime     合理的文件时间
   * @return 新文件的路径
   */
  def generateInputPathsBySize(conf: Configuration, trainDataPath: String,
                               totalFileSize: Long, validTime: Long): String = {

    val pathBuffer = new ListBuffer[String]
    val path = new Path(trainDataPath)
    val fileSystem = path.getFileSystem(conf)

    // 读取以日期为单位的目录
    val fileList = fileSystem.globStatus(path)
    util.Arrays.sort(fileList, new Comparator[FileStatus]() {
      override def compare(lhs: FileStatus, rhs: FileStatus): Int = {
        if (rhs.getModificationTime == lhs.getModificationTime)
          return 0
        else if (rhs.getModificationTime > lhs.getModificationTime)
          return 1
        else
          return -1
      }
    })

    var itemIndex = 0
    var fileSize = 0L
    var foundIt = false
    while (itemIndex < fileList.length && !foundIt) {
      val item = fileList(itemIndex)
      if (item.isFile()) {
        if (fileSize < totalFileSize) {
          if (item.getModificationTime() < validTime) {
            pathBuffer.append(item.getPath().toString())
            fileSize += item.getLen()
          } else {
            // 文件修改时间不满足，寻找下一个
          }
        } else {
          foundIt = true
        }
      }

      itemIndex += 1
    }

    val pathArray = pathBuffer.toArray
    if (pathArray.length == 0) {
      return ""
    } else if (pathArray.length == 1) {
      return pathArray(0)
    } else {
      return pathArray.mkString(",")
    }
  }

  /**
   * 删除 HDFS 上的文件或者目录
   *
   * @param sc   Spark环境变量
   * @param src  HDFS文件或者目录地址
   * @param user HDFS用户名和用户组，比如 lhotse,supergroup
   */
  def delete(sc: SparkContext, src: String, user: String = null): Unit = {
    Try {
      val HDFS = "hdfs://" + src.split("/")(2)
      val fs = if (user != null && user.nonEmpty) FileSystem.get(URI.create(HDFS), sc
        .hadoopConfiguration, user)
      else FileSystem.get(URI.create(HDFS), sc.hadoopConfiguration)
      val srcpath = new Path(src)
      if (fs.exists(srcpath)) fs.delete(srcpath, true)
    }.getOrElse(println(s"cannot delete $src"))
  }

  /**
   * 检查HDFS地址是否存在
   *
   * @param sc   Spark环境变量
   * @param src  HDFS文件或者目录地址
   * @param user HDFS用户名和用户组，比如 lhotse,supergroup
   * @return true: 存在，false: 不存在
   */
  def exists(sc: SparkContext, src: String, user: String = null): Boolean = {
    val HDFS = "hdfs://" + src.split("/")(2)
    val fs = if (user != null && user.nonEmpty) FileSystem.get(URI.create(HDFS), sc
      .hadoopConfiguration, user)
    else FileSystem.get(URI.create(HDFS), sc.hadoopConfiguration)
    val srcpath = new Path(src)
    fs.exists(srcpath)
  }
}
