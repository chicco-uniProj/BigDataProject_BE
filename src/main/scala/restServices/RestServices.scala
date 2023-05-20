package restServices

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.fpm.{FPGrowth, FPGrowthModel}



object RestServices extends cask.MainRoutes {
  override def port: Int = 6969
  val ss = SparkSession
    .builder()
    .master("local[*]")
    .appName("PippoFranco")
    .getOrCreate()
  import ss.implicits._
  val sc = ss.sparkContext
  val prodotti = ss.read.options(Map("header" -> "true", "inferSchema" -> "true")).csv("src/main/data/InstaCartOnlineGrocery/products.csv")
  val dipartimenti = ss.read.options(Map("header" -> "true", "inferSchema" -> "true")).csv("src/main/data/InstaCartOnlineGrocery/departments.csv")
  val corsie = ss.read.options(Map("header" -> "true", "inferSchema" -> "true")).csv("src/main/data/InstaCartOnlineGrocery/aisles.csv")
  val ordiniProdotti=ss.read.options(Map("header"->"true","inferSchema"->"true")).csv("src/main/data/InstaCartOnlineGrocery/order_products__prior.csv").union(ss.read.options(Map("inferSchema"->"true","headers"->"true")).csv("src/main/data/InstaCartOnlineGrocery/order_products__train.csv"))
  val infoOrdini=ss.read.options(Map("header"->"true","inferSchema"->"true")).csv("src/main/data/InstaCartOnlineGrocery/orders.csv")
  sc.setLogLevel("ERROR")
  var modello :FPGrowthModel=null

  @cask.get("/show")
  def showData()={
    prodotti.show()
    prodotti.printSchema()
    dipartimenti.show()
    dipartimenti.printSchema()
    corsie.show()
    corsie.printSchema()
    ordiniProdotti.show()
    ordiniProdotti.printSchema()
    infoOrdini.show()
    infoOrdini.printSchema()
  }

  //topXParametri da risolvere il parametro nell'url
  @cask.get("/topProdotti")
  def topXProdotti2(numProdotti:Int): String={
    val numeroRighe=ordiniProdotti.count()
    println(numeroRighe)
    val ordiniGroupati=ordiniProdotti.groupBy("product_id").count().sort(col("count").desc)
    val topNOrdini=ordiniGroupati.limit(numProdotti).join(prodotti,ordiniGroupati("product_id") === prodotti("product_id")).select("product_name","count").sort(col("count").desc)
    val numProdottiTop=topNOrdini.agg(sum("count")).as[Long].collect()(0)
    println(numProdottiTop)
    val nuovoDF=(Seq(("others",numeroRighe-numProdottiTop)).toDF("product_name","count"))
    val finale=topNOrdini.union(nuovoDF)
    topNOrdini.show()
    finale.show()
    val risposta=creaRisposta(finale)
    println(risposta)
    return risposta
  }

  @cask.get("/oreGiornoPiuAcquisti")
  def oreGiornoPiuAcquisti(): String ={
    val finale=infoOrdini.groupBy("order_hour_of_day").count().sort(col("order_hour_of_day").asc)
    val risposta=creaRisposta(finale)
    return risposta
  }

  @cask.get("/giorniDaAcquistoPrecedente")
  def giorniDaAcquistoPrec(): String ={
    val finale=infoOrdini.groupBy("days_since_prior_order").count().sort(col("days_since_prior_order").asc).na.drop()
    finale.show()
    val risposta=creaRisposta(finale)
    return risposta
  }

  @cask.get("/giorniDellaSettimana")
  def giorniDellaSettimana(): String={
    val finale=infoOrdini.groupBy("order_dow").count().sort(col("order_dow").asc)
    val risposta=creaRisposta(finale)
    return risposta
  }

  @cask.get("/numeroOggettiInOrdine")
  def numeroOggettiInOrdine(): String={
    val ordineNumprodotti=ordiniProdotti.join(prodotti,ordiniProdotti("product_id") === prodotti("product_id")).groupBy("order_id").count().sort(col("count").desc)
    ordineNumprodotti.createOrReplaceTempView("ordineNumprodotti")
    val ordineNumProdottiAlias=ss.sql("SELECT order_id,count AS numProdotti FROM ordineNumprodotti")
    val finale=ordineNumProdottiAlias.groupBy("numProdotti").count().sort(col("numProdotti").asc)
    finale.show()
    val risposta=creaRisposta(finale)
    return risposta
  }

  @cask.get("/dipartimentiProdotti")
  def dipartimentiProdottiVenduti(): String={
    val dfSemiJoinato=ordiniProdotti.join(prodotti,ordiniProdotti("product_id")===prodotti("product_id"))
    val dfJoinato=dfSemiJoinato.join(dipartimenti,dipartimenti("department_id")===dfSemiJoinato("department_id"))
    val finale=dfJoinato.groupBy("department").agg(count("*").as("numProdottiVenduti")).sort(col("numProdottiVenduti").desc)
    finale.show()
    val risposta=creaRisposta(finale)
    return risposta
  }

  @cask.get("/regoleAssociazione") //confidence lift support
  def regoleAssociazione(sortBy:String,limit:Int): String={
    val ordiniTrain = ss.read.options(Map("header" -> "true", "inferSchema" -> "true")).csv("src/main/data/InstaCartOnlineGrocery/order_products__train.csv")
    ordiniTrain.show()
    ordiniTrain.printSchema()
    prodotti.printSchema()
    val ordiniTrainProdotti=ordiniTrain.join(prodotti,ordiniTrain("product_id") === prodotti("product_id"))
    val basket=ordiniTrainProdotti.groupBy("order_id").agg(collect_set("product_name").alias("items"))
    basket.createOrReplaceTempView("baskets")
    val baskets_ds = ss.sql("select items from baskets").toDF("items")
    val fpgrowth=new FPGrowth().setItemsCol("items").setMinSupport(0.001).setMinConfidence(0)
    modello=fpgrowth.fit(baskets_ds)
    val oggettiPiuPopolariNelBasket=modello.freqItemsets
    oggettiPiuPopolariNelBasket.show()
    modello.associationRules.show()
    val ifThen = modello.associationRules.sort(col(sortBy).desc).limit(limit)//regole di associazione
    val risposta=creaRisposta(ifThen)
    return risposta
  }

  @cask.get("/prodottiCompratiInsieme")
  def prodottiCompratiInsieme(numProdotti:Int): String={
    val ordiniTrain = ss.read.options(Map("header" -> "true", "inferSchema" -> "true")).csv("src/main/data/InstaCartOnlineGrocery/order_products__train.csv")
    ordiniTrain.show()
    ordiniTrain.printSchema()
    prodotti.printSchema()
    val ordiniTrainProdotti=ordiniTrain.join(prodotti,ordiniTrain("product_id") === prodotti("product_id"))
    val basket=ordiniTrainProdotti.groupBy("order_id").agg(collect_set("product_name").alias("items"))
    basket.createOrReplaceTempView("baskets")
    val baskets_ds = ss.sql("select items from baskets").toDF("items")
    val fpgrowth=new FPGrowth().setItemsCol("items").setMinSupport(0.001).setMinConfidence(0)
    val modello=fpgrowth.fit(baskets_ds)
    val oggettiPiuPopolariNelBasket=modello.freqItemsets.sort(col("freq").desc).where("size(items)>="+numProdotti)
    oggettiPiuPopolariNelBasket.show()
    return creaRisposta(oggettiPiuPopolariNelBasket)
  }

  def creaRisposta(dati: DataFrame): String ={
    val ciao=Array("Paolo","Fabrizio")
    val array=dati.toJSON.collectAsList().toArray(ciao)
    val sb:StringBuilder=new StringBuilder()
    sb.append("[")
    array.foreach({
      case(x)=>sb.append(x)
        sb.append(",")
    })
    sb.deleteCharAt(sb.length()-1)
    sb.append("]")
    return sb.toString()
  }


  @cask.get("/predizione")
  def predizione(dati:Seq[String]):String={

    print(dati)
    val ifThen=modello.associationRules
    //se trovo una testa di una regola di associazione i cui elementi sono contenuti
    //tutti quanti nello scontrino che sto osservando, predico che la coda sia un buon consiglio.
    val proposte=ifThen.filter({x=>x.getAs[scala.collection.mutable.Seq[String]]("antecedent").forall(dati.contains)})
    val finale=proposte.sort(col("confidence").desc)./*filter({x => x.getAs[scala.collection.mutable.Seq[String]]("consequent").forall(!dati.contains)}).*/limit(5)
    finale.show()
    val proposals = finale.rdd.map{ rule =>
      val antecedent = rule.getAs[scala.collection.mutable.Seq[String]]("antecedent")
      val consequent = rule.getAs[scala.collection.mutable.Seq[String]]("consequent")
      if (antecedent.forall(dati.contains)) {
        consequent.toSet
      } else {
        Set.empty[String]
      }
    }.reduce(_ ++ _)

    val nonTrivialProposals = proposals.filterNot(dati.contains)
    val sb:StringBuilder=new StringBuilder()
    sb.append("{\"suggestions\":")
    sb.append("[")
    for (t <- nonTrivialProposals){
      sb.append("\""+t+"\"")
      sb.append(",")
    }
    sb.deleteCharAt(sb.length()-1)
    sb.append("]}")
    return sb.toString()
  }
  initialize()
}