package dm.sns.utils

import org.apache.log4j._

object Logging {

  def loggingConfig(clazz: Class[_]): Unit = {
    val rootLogger = Logger.getRootLogger
    rootLogger.setLevel(Level.ERROR)

    val myClassLogger = Logger.getLogger(clazz)
    myClassLogger.setLevel(Level.INFO)

    val consoleAppender = new ConsoleAppender()
    consoleAppender.setLayout(new PatternLayout("%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n"))
    consoleAppender.activateOptions()

    myClassLogger.addAppender(consoleAppender)
  }
}
