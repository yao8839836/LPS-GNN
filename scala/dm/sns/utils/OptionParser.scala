package dm.sns.utils

class OptionParser(args: Array[String]) {

  private val options = args
    .map(arg => arg.split("="))
    .filter(_.size == 2)
    .map(kv => kv(0).trim() -> kv(1).trim())
    .toMap

  def getInt(key: String): Int = options.get(key) match {
    case Some(v) => v.toInt
    case None    => throw new IllegalArgumentException(s"$key is required!")
  }

  def getInt(key: String, default: Int): Int = options.get(key) match {
    case Some(v) => v.toInt
    case None    => default
  }

  def getBoolean(key: String): Boolean = options.get(key) match {
    case Some(v) => v.toBoolean
    case None    => throw new IllegalArgumentException(s"$key is required!")
  }

  def getBoolean(key: String, default: Boolean): Boolean = options.get(key) match {
    case Some(v) => v.toBoolean
    case None    => default
  }

  def getLong(key: String): Long = options.get(key) match {
    case Some(v) => v.toLong
    case None    => throw new IllegalArgumentException(s"$key is required!")
  }

  def getLong(key: String, default: Long): Long = options.get(key) match {
    case Some(v) => v.toLong
    case None    => default
  }

  def getDouble(key: String): Double = options.get(key) match {
    case Some(v) => v.toDouble
    case None    => throw new IllegalArgumentException(s"$key is required!")
  }

  def getDouble(key: String, default: Double): Double = options.get(key) match {
    case Some(v) => v.toDouble
    case None    => default
  }

  def getString(key: String): String = options.get(key) match {
    case Some(v) => v
    case None    => throw new IllegalArgumentException(s"$key is required!")
  }

  def getString(key: String, default: String): String = options.get(key) match {
    case Some(v) => v
    case None    => default
  }

  def getStringMap(key: String): Map[String, String] = getStringMap(key, ",", ";")

  def getStringMap(key: String, kvSplitOp: String, itemSplitOp: String): Map[String, String] = options.get(key) match {
    case Some(v) =>
      v
        .split(s"\\s*$itemSplitOp\\s*")
        .map(item => {
          val kv = item.split(s"\\s*$kvSplitOp\\s*")
          kv(0) -> kv(1)
        })
        .toMap
    case None => throw new IllegalArgumentException(s"$key is required!")
  }

  def getStringMap(key: String, default: String): Map[String, String] = getStringMap(key, default, ",", ";")

  def getStringMap(key: String, default: String, kvSplitOp: String, itemSplitOp: String): Map[String, String] =
    options.get(key) match {
      case Some(v) =>
        v
          .split(s"\\s*$itemSplitOp\\s*")
          .map(item => {
            val kv = item.split(s"\\s*$kvSplitOp\\s*")
            kv(0) -> kv(1)
          })
          .toMap
      case None =>
        default
          .split(s"\\s*$itemSplitOp\\s*")
          .map(item => {
            val kv = item.split(s"\\s*$kvSplitOp\\s*")
            kv(0) -> kv(1)
          })
          .toMap
    }
}
